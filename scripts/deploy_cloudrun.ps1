<#
.SYNOPSIS
    Build Anime Compass with Cloud Build and deploy that exact image to Cloud Run.

.DESCRIPTION
    Reads the active project from gcloud, checks that the one-time setup in
    docs/CLOUD_RUN_DEPLOYMENT.md has been done, builds an image tagged with the
    current short commit SHA, deploys it by digest, and prints the service URLs.

    Every check that can fail runs before the build, so a missing repository,
    service account or secret costs seconds rather than a wasted build.

    Stops at the first failure. Never reads or prints the Gemini API key: the
    service references it in Secret Manager by name and pinned version only.

.PARAMETER GeminiSecretVersion
    The version of the gemini-api-key secret exposed as GEMINI_API_KEY. A
    positive integer; "latest" is refused. When rotating the key, bump the
    default in the param block below and commit, so the version production runs
    is recorded alongside the code that deploys it.

.PARAMETER AllowDirty
    Deploy even though the working tree has uncommitted changes. The image is
    built from the files on disk, so without this switch the commit SHA in the
    tag could describe code that is not what shipped. With it, the tag carries
    a -dirty suffix so the image says so.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1 -GeminiSecretVersion 2
#>
[CmdletBinding()]
param(
    # Pinned rather than "latest". Cloud Run resolves a secret exposed as an
    # environment variable when each instance starts, so "latest" would let a
    # rotated key reach new instances while running ones keep the old value,
    # and make which key a revision uses depend on when it happened to start.
    # Google recommends pinning a version for secrets used this way.
    [ValidatePattern('^[1-9][0-9]*$')]
    [string]$GeminiSecretVersion = '1',

    [switch]$AllowDirty
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Region = 'us-central1'
$Repository = 'anime-compass'
$Service = 'anime-compass'
$ImageName = 'anime-compass'
$RuntimeAccountName = 'anime-compass-run'
$SecretName = 'gemini-api-key'
$HfDatasetRepo = 'Seiitsu/anime-compass-data'
$ContainerPort = 8080

# Requests one instance handles at once. Cloud Run's default is 80; this service
# has 1 vCPU, keeps its ML stack in memory and runs at most one instance, so 4
# is a conservative first-production value. Tune it from Cloud Run's latency,
# CPU and memory metrics rather than by guessing.
$Concurrency = 4

# Cloud Run keeps traffic away until /api/ready returns 200. A 503 while the
# recommender initializes is intended and simply counts as "not yet"; the
# budget is 2 + (2 x 60) = 122 seconds.
$StartupProbe = "httpGet.path=/api/ready,httpGet.port=$ContainerPort,initialDelaySeconds=2,timeoutSeconds=2,periodSeconds=2,failureThreshold=60"


function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}


function Resolve-Tool {
    # gcloud.cmd is preferred over gcloud.ps1 so exit codes come back through a
    # single, predictable path.
    param([Parameter(Mandatory = $true)][string[]]$Names)
    foreach ($name in $Names) {
        $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $command) {
            return $command.Path
        }
    }
    throw "Required tool not found on PATH: $($Names -join ' or ')"
}


function Invoke-Tool {
    # Run a native tool with its output streaming to the console.
    #
    # Windows PowerShell 5.1 does not turn a native command's non-zero exit code
    # into a terminating error, even under ErrorActionPreference=Stop, so it is
    # checked explicitly. The preference drops to Continue for the call itself
    # because gcloud writes progress to stderr, which 5.1 can otherwise surface
    # as an error record and abort on a perfectly successful command.
    param(
        [Parameter(Mandatory = $true)][string]$Tool,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Tool @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
}


function Get-ToolOutput {
    # Same contract as Invoke-Tool, but captures stdout and returns it trimmed.
    #
    # Callers ask gcloud for --format=json and parse it rather than using
    # --format=value(...). gcloud.cmd runs through cmd.exe, and unquoted
    # parentheses are exactly the kind of argument cmd.exe can mangle.
    param(
        [Parameter(Mandatory = $true)][string]$Tool,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Tool @Arguments 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
    return ((@($output) -join "`n").Trim())
}


$failed = $false
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $RepoRoot
try {
    Write-Step 'Checking tools'
    $gcloud = Resolve-Tool -Names @('gcloud.cmd', 'gcloud')
    $git = Resolve-Tool -Names @('git.exe', 'git')
    Write-Host "  gcloud  $gcloud"
    Write-Host "  git     $git"

    Write-Step 'Reading the active Google Cloud project'
    $project = Get-ToolOutput -Tool $gcloud -Arguments @('config', 'get-value', 'project') `
        -Description 'Reading the active project'
    if ([string]::IsNullOrWhiteSpace($project) -or $project -eq '(unset)') {
        throw 'No active Google Cloud project. Run: gcloud config set project YOUR_PROJECT_ID'
    }
    Write-Host "  project $project"

    Write-Step 'Identifying the commit being deployed'
    $sha = Get-ToolOutput -Tool $git -Arguments @('rev-parse', '--short', 'HEAD') `
        -Description 'Reading the commit SHA'
    $changes = Get-ToolOutput -Tool $git -Arguments @('status', '--porcelain') `
        -Description 'Checking the working tree'
    $tag = $sha
    if (-not [string]::IsNullOrWhiteSpace($changes)) {
        if (-not $AllowDirty) {
            throw ("The working tree has uncommitted changes, so commit $sha would not describe " +
                'what gets built. Commit first, or re-run with -AllowDirty.')
        }
        $tag = "$sha-dirty"
        Write-Warning "Deploying uncommitted changes; the image is tagged $tag."
    }

    $runtimeAccount = "$RuntimeAccountName@$project.iam.gserviceaccount.com"
    $imageRepository = "$Region-docker.pkg.dev/$project/$Repository/$ImageName"
    $image = "${imageRepository}:$tag"
    Write-Host "  commit  $sha"
    Write-Host "  image   $image"

    Write-Step 'Checking that this gcloud can configure a startup probe'
    $deployHelp = Get-ToolOutput -Tool $gcloud -Arguments @('run', 'deploy', '--help') `
        -Description 'Reading gcloud run deploy help'
    if ($deployHelp -notmatch '--startup-probe') {
        throw ("This gcloud does not offer --startup-probe on 'gcloud run deploy'. " +
            "Update it with 'gcloud components update' and re-run. Stopped before building.")
    }

    Write-Step 'Checking the one-time setup'
    [void](Get-ToolOutput -Tool $gcloud `
            -Arguments @('artifacts', 'repositories', 'describe', $Repository, "--location=$Region", "--project=$project", '--format=json') `
            -Description "Finding Artifact Registry repository '$Repository' in $Region")
    Write-Host "  repository       $Repository"
    [void](Get-ToolOutput -Tool $gcloud `
            -Arguments @('iam', 'service-accounts', 'describe', $runtimeAccount, "--project=$project", '--format=json') `
            -Description "Finding runtime service account $runtimeAccount")
    Write-Host "  service account  $runtimeAccount"
    # Version metadata only. Nothing here can read the secret's value. Checked
    # before building because a pinned version that is missing, disabled or
    # destroyed would otherwise fail the deploy only after a full build.
    $versionJson = Get-ToolOutput -Tool $gcloud `
        -Arguments @('secrets', 'versions', 'describe', $GeminiSecretVersion, "--secret=$SecretName", "--project=$project", '--format=json') `
        -Description "Finding version $GeminiSecretVersion of Secret Manager secret '$SecretName'"
    $versionState = ($versionJson | ConvertFrom-Json).state
    if ($versionState -ne 'ENABLED') {
        throw ("Version $GeminiSecretVersion of secret '$SecretName' is $versionState, not ENABLED. " +
            'Pin an enabled version with -GeminiSecretVersion.')
    }
    Write-Host "  secret           ${SecretName} version $GeminiSecretVersion ($versionState)"

    Write-Step "Building $image with Cloud Build"
    Invoke-Tool -Tool $gcloud `
        -Arguments @('builds', 'submit', '.', '--config=cloudbuild.cloudrun.yaml', "--substitutions=_IMAGE=$image", "--project=$project") `
        -Description 'Cloud Build'

    Write-Step 'Resolving the pushed image digest'
    # Tags can be moved; a digest cannot. Deploying by digest guarantees Cloud
    # Run runs the bytes this build produced.
    $deployImage = $image
    try {
        $imageJson = Get-ToolOutput -Tool $gcloud `
            -Arguments @('artifacts', 'docker', 'images', 'describe', $image, "--project=$project", '--format=json') `
            -Description 'Reading the pushed image'
        $digestReference = ($imageJson | ConvertFrom-Json).image_summary.fully_qualified_digest
        if (-not [string]::IsNullOrWhiteSpace($digestReference)) {
            $deployImage = $digestReference
        }
    }
    catch {
        Write-Warning "Could not read the image digest ($($_.Exception.Message)); deploying the tag this build just pushed."
    }
    Write-Host "  deploying $deployImage"

    Write-Step "Deploying $Service to Cloud Run in $Region"
    $deployArguments = @(
        'run', 'deploy', $Service,
        "--image=$deployImage",
        "--project=$project",
        "--region=$Region",
        "--service-account=$runtimeAccount",
        '--cpu=1',
        '--memory=4Gi',
        '--min-instances=0',
        '--max-instances=1',
        '--cpu-boost',
        "--port=$ContainerPort",
        "--concurrency=$Concurrency",
        "--startup-probe=$StartupProbe",
        "--set-env-vars=LLM_PROVIDER=gemini,HF_DATASET_REPO=$HfDatasetRepo",
        "--set-secrets=GEMINI_API_KEY=${SecretName}:$GeminiSecretVersion",
        '--allow-unauthenticated',
        '--quiet'
    )
    Invoke-Tool -Tool $gcloud -Arguments $deployArguments -Description 'Cloud Run deployment'

    Write-Step 'Ensuring public unauthenticated access'
    # --allow-unauthenticated can leave only a warning when an organization
    # policy forbids public access. Granting the binding explicitly turns that
    # into a failure instead of a service that is quietly private.
    [void](Get-ToolOutput -Tool $gcloud `
            -Arguments @('run', 'services', 'add-iam-policy-binding', $Service, "--project=$project", "--region=$Region", '--member=allUsers', '--role=roles/run.invoker', '--format=json') `
            -Description 'Granting allUsers the Cloud Run invoker role')
    Write-Host '  allUsers -> roles/run.invoker'

    $serviceJson = Get-ToolOutput -Tool $gcloud `
        -Arguments @('run', 'services', 'describe', $Service, "--project=$project", "--region=$Region", '--format=json') `
        -Description 'Reading the service URL'
    $described = $serviceJson | ConvertFrom-Json
    $url = $described.status.url
    if ([string]::IsNullOrWhiteSpace($url)) {
        throw 'Cloud Run reported no service URL.'
    }

    Write-Host ''
    Write-Host 'Cloud Run reports the revision is serving.' -ForegroundColor Green
    Write-Host "  Service    $url"
    Write-Host "  Health     $url/api/health"
    Write-Host "  Ready      $url/api/ready"
    Write-Host "  Revision   $($described.status.latestReadyRevisionName)"
    Write-Host "  Image      $deployImage"
    Write-Host ''
    Write-Host 'After the service scales to zero, the next request waits while models load;'
    Write-Host '/api/ready answers 503 service_warming until the instance is serving again.'
}
catch {
    $failed = $true
    Write-Host ''
    Write-Host "Deployment stopped: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Pop-Location
}

if ($failed) {
    exit 1
}
