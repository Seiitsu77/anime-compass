const COMPACT_RESULT_LIMIT = 12;
const CATALOG_PAGE_SIZE = 24;
const SESSION_STORAGE_KEY = "animeCompassSession";
let searchRequestSequence = 0;

const state = {
  agentOpen: false,
  agentGreeted: false,
  liked: [],
  preferences: {
    disliked_titles: [],
    seen_titles: [],
    excluded_titles: [],
    preferred_genres: [],
    excluded_genres: [],
    temporary_ratings: {},
    previous_reference_titles: [],
  },
  sessionId: "",
  currentResults: new Map(),
  allGenres: [],
  filterOpen: false,
  genres: new Set(),
  excludedGenres: new Set(),
  genreCovers: {},
  mediaType: "",
  studio: "",
  minScore: "",
  maxEpisodes: "",
  minYear: "",
  maxYear: "",
  sortBy: "relevance",
  catalogOffset: 0,
  catalogTotal: 0,
  searchQuery: "",
  history: [],
  lastTrace: [],
};

const els = {
  activeFilters: document.querySelector("#activeFilters"),
  agentDock: document.querySelector("#agentDock"),
  agentForm: document.querySelector("#agentForm"),
  agentInput: document.querySelector("#agentInput"),
  agentStarters: document.querySelector("#agentStarters"),
  agentStatus: document.querySelector("#agentStatus"),
  agentToggle: document.querySelector("#agentToggle"),
  catalogStatus: document.querySelector("#catalogStatus"),
  chatLog: document.querySelector("#chatLog"),
  closeAgentBtn: document.querySelector("#closeAgentBtn"),
  closeFiltersBtn: document.querySelector("#closeFiltersBtn"),
  clearGenresBtn: document.querySelector("#clearGenresBtn"),
  clearTasteBtn: document.querySelector("#clearTasteBtn"),
  categoryCount: document.querySelector("#categoryCount"),
  detailContent: document.querySelector("#detailContent"),
  detailModal: document.querySelector("#detailModal"),
  filterPanel: document.querySelector("#filterPanel"),
  filterSummary: document.querySelector("#filterSummary"),
  filterToggle: document.querySelector("#filterToggle"),
  genreChips: document.querySelector("#genreChips"),
  likedCount: document.querySelector("#likedCount"),
  likedList: document.querySelector("#likedList"),
  loadMoreBtn: document.querySelector("#loadMoreBtn"),
  maxEpisodesInput: document.querySelector("#maxEpisodesInput"),
  maxYearInput: document.querySelector("#maxYearInput"),
  minScoreInput: document.querySelector("#minScoreInput"),
  minYearInput: document.querySelector("#minYearInput"),
  recommendationGrid: document.querySelector("#recommendationGrid"),
  resultsTitle: document.querySelector("#resultsTitle"),
  resetSessionBtn: document.querySelector("#resetSessionBtn"),
  pageBackdrop: document.querySelector("#pageBackdrop"),
  searchInput: document.querySelector("#searchInput"),
  searchResults: document.querySelector("#searchResults"),
  sortSelect: document.querySelector("#sortSelect"),
  studioInput: document.querySelector("#studioInput"),
  topRecommendBtn: document.querySelector("#topRecommendBtn"),
  traceOutput: document.querySelector("#traceOutput"),
  typeSelect: document.querySelector("#typeSelect"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatScore(score) {
  return score ? Number(score).toFixed(2) : "N/A";
}

function formatMeta(item) {
  const parts = [];
  if (item.type) parts.push(item.type);
  if (item.start_year) parts.push(item.start_year);
  if (item.episodes) parts.push(`${item.episodes} eps`);
  parts.push(`Score ${formatScore(item.score)}`);
  return parts.join(" / ");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    const detail = typeof payload.error === "object" ? payload.error?.message : payload.error;
    throw new Error(detail || "Request failed");
  }
  return payload;
}

function debounce(fn, delay = 220) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

function createSessionId() {
  if (crypto?.randomUUID) return crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function defaultPreferences() {
  return {
    disliked_titles: [],
    seen_titles: [],
    excluded_titles: [],
    preferred_genres: [],
    excluded_genres: [],
    temporary_ratings: {},
    previous_reference_titles: [],
  };
}

function loadSessionState() {
  try {
    const saved = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || "{}");
    state.sessionId = saved.sessionId || createSessionId();
    state.liked = Array.isArray(saved.liked) ? saved.liked : [];
    state.preferences = { ...defaultPreferences(), ...(saved.preferences || {}) };
  } catch {
    state.sessionId = createSessionId();
    state.preferences = defaultPreferences();
  }
  saveSessionState();
}

function saveSessionState() {
  localStorage.setItem(
    SESSION_STORAGE_KEY,
    JSON.stringify({
      sessionId: state.sessionId,
      liked: state.liked,
      preferences: state.preferences,
    }),
  );
}

function preferencePayload() {
  return {
    liked_titles: state.liked.map((item) => item.title),
    ...state.preferences,
  };
}

async function syncSessionProfile() {
  if (!state.sessionId) return;
  await api("/api/session/preferences", {
    method: "POST",
    body: JSON.stringify({
      session_id: state.sessionId,
      replace: true,
      ...preferencePayload(),
    }),
  });
}

function rememberPreference(field, title) {
  if (!title) return;
  const values = state.preferences[field] || [];
  if (!values.some((value) => value.toLowerCase() === title.toLowerCase())) {
    values.push(title);
  }
  state.preferences[field] = values;
  saveSessionState();
}

async function initialize() {
  loadSessionState();
  renderLiked();
  try {
    const health = await api("/api/health");
    const meta = health.catalog;

    state.allGenres = meta.genres || [];
    state.genreCovers = meta.genre_covers || {};
    els.catalogStatus.textContent = `${meta.count.toLocaleString()} titles loaded`;
    els.catalogStatus.className = "status-pill online";
    updateAgentStatus(health.agent);
    renderTypeOptions(meta.types);
    renderGenreChips(meta.genres);
    renderPageBackdrop();
    await refreshRecommendations();
  } catch (error) {
    els.catalogStatus.textContent = "Catalog failed to load";
    els.catalogStatus.className = "status-pill offline";
    els.recommendationGrid.innerHTML = `<div class="loading">${escapeHtml(error.message)}</div>`;
  }
}

function updateAgentStatus(agent) {
  if (agent?.available) {
    const provider = agent.provider ? `${agent.provider} / ` : "";
    els.agentStatus.textContent = `${provider}${agent.model || "agent"} online`;
    els.agentStatus.className = "status-pill online";
  } else {
    els.agentStatus.textContent = "Catalog tools ready / LLM offline";
    els.agentStatus.className = "status-pill degraded";
  }
}

function renderTypeOptions(types) {
  for (const type of types || []) {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    els.typeSelect.appendChild(option);
  }
}

function formatFilterSummary() {
  const categoryCount = state.genres.size + state.excludedGenres.size;
  const categoryText = `${categoryCount} ${categoryCount === 1 ? "category" : "categories"}`;
  const formatText = state.mediaType || "Any format";
  return `${categoryText} / ${formatText}`;
}

function hasActiveFilters() {
  return Boolean(
    state.genres.size ||
      state.excludedGenres.size ||
      state.mediaType ||
      state.studio ||
      state.minScore ||
      state.maxEpisodes ||
      state.minYear ||
      state.maxYear ||
      state.sortBy !== "relevance"
  );
}

function updateResultsTitle(count = null) {
  const suffix = count === null ? "" : ` (${Number(count).toLocaleString()})`;

  if (state.searchQuery) {
    els.resultsTitle.textContent = `Search results for "${state.searchQuery}"${suffix}`;
    return;
  }

  if (hasActiveFilters()) {
    els.resultsTitle.textContent = `All matching titles${suffix}`;
    return;
  }

  els.resultsTitle.textContent = state.liked.length ? "Personalized for your taste" : "Start with highly rated picks";
}

function setFilterOpen(open) {
  state.filterOpen = open;
  els.filterPanel.classList.toggle("is-open", open);
  els.filterToggle.setAttribute("aria-expanded", String(open));
  els.filterPanel.setAttribute("aria-hidden", String(!open));
}

function setAgentOpen(open) {
  state.agentOpen = open;
  els.agentDock.classList.toggle("is-hidden", !open);
  els.agentDock.setAttribute("aria-hidden", String(!open));
  els.agentToggle.hidden = open;

  if (open) {
    if (!state.agentGreeted) {
      addMessage(
        "assistant",
        "Hi, I'm Compass. I can introduce a title without spoilers, explore its cast and staff, or build a recommendation list around your exact constraints.",
      );
      state.agentGreeted = true;
    }
    requestAnimationFrame(() => els.agentInput.focus());
  }
}

function renderGenreChips(genres) {
  const categoryCount = state.genres.size + state.excludedGenres.size;
  els.categoryCount.textContent = `${categoryCount} selected`;
  els.filterSummary.textContent = formatFilterSummary();
  els.genreChips.innerHTML = (genres || [])
    .filter(Boolean)
    .map((genre) => {
      const included = state.genres.has(genre);
      const excluded = state.excludedGenres.has(genre);
      const status = included ? "active" : excluded ? "excluded" : "";
      const action = included ? "Included" : excluded ? "Excluded" : "+";
      return `<button class="genre-chip ${status}" type="button" data-genre="${escapeHtml(genre)}" aria-label="${escapeHtml(genre)} filter: ${action}">
        <span>${escapeHtml(genre)}</span>
        <span>${action}</span>
      </button>`;
    })
    .join("");
}

function renderPageBackdrop() {
  const selected = Array.from(state.genres);
  const genreNames = selected.length ? selected : Object.keys(state.genreCovers);
  const baseUrls = genreNames.map((genre) => state.genreCovers[genre]).filter(Boolean);
  const urls = [...baseUrls, ...baseUrls, ...baseUrls].slice(0, 32);

  els.pageBackdrop.innerHTML = urls
    .map((url) => `<img src="${escapeHtml(url)}" alt="" loading="lazy" />`)
    .join("");
}

async function searchAnime(query, allowEmpty = false) {
  const requestSequence = ++searchRequestSequence;
  if (state.searchQuery && query === state.searchQuery) {
    els.searchResults.innerHTML = "";
    return;
  }
  if (!query && !allowEmpty) {
    els.searchResults.innerHTML = "";
    return;
  }

  const payload = await api(`/api/anime/search?q=${encodeURIComponent(query)}&limit=8`);
  if (requestSequence !== searchRequestSequence) return;
  renderSearchResults(payload.results);
}

async function submitSearch(query) {
  searchRequestSequence += 1;
  state.searchQuery = query.trim();
  els.searchResults.innerHTML = "";

  if (!state.searchQuery) {
    await refreshRecommendations();
    return;
  }

  await loadCatalogPage(false);
}

function renderSearchResults(results) {
  if (!results.length) {
    els.searchResults.innerHTML = `<div class="empty-state">No matches found.</div>`;
    return;
  }

  els.searchResults.innerHTML = results
    .map((item) => {
      const alreadyLiked = state.liked.some((liked) => liked.id === item.id);
      return `
        <article class="search-result">
          <img class="poster-thumb" src="${escapeHtml(item.image_url || "")}" alt="" loading="lazy" />
          <div>
            <p class="item-title">${escapeHtml(item.title)}</p>
            <p class="item-meta">${escapeHtml(formatMeta(item))}</p>
          </div>
          <button class="secondary-button" type="button" data-detail-id="${item.id}">Details</button>
          <button class="icon-button" type="button" data-add-id="${item.id}" aria-label="Add ${escapeHtml(item.title)} to liked titles" ${alreadyLiked ? "disabled" : ""}>+</button>
        </article>
      `;
    })
    .join("");
}

function addLiked(item) {
  if (state.liked.some((liked) => liked.id === item.id)) return;
  state.liked.push(item);
  saveSessionState();
  syncSessionProfile().catch(console.error);
  renderLiked();
  refreshRecommendations();
}

function removeLiked(id) {
  state.liked = state.liked.filter((item) => item.id !== Number(id));
  saveSessionState();
  syncSessionProfile().catch(console.error);
  renderLiked();
  refreshRecommendations();
}

function renderLiked() {
  els.likedCount.textContent = String(state.liked.length);

  if (!state.liked.length) {
    els.likedList.className = "liked-list empty-state";
    els.likedList.textContent = "Add titles from search or recommendations.";
    return;
  }

  els.likedList.className = "liked-list";
  els.likedList.innerHTML = state.liked
    .map(
      (item) => `
        <article class="liked-item">
          <span class="item-title">${escapeHtml(item.title)}</span>
          <button class="icon-button remove" type="button" data-remove-id="${item.id}">x</button>
        </article>
      `,
    )
    .join("");
}

function renderActiveFilters() {
  const filters = Array.from(state.genres);
  filters.push(...Array.from(state.excludedGenres).map((genre) => `Not ${genre}`));
  if (state.mediaType) filters.push(state.mediaType);
  if (state.studio) filters.push(`Studio: ${state.studio}`);
  if (state.minScore) filters.push(`Score ${state.minScore}+`);
  if (state.maxEpisodes) filters.push(`${state.maxEpisodes} eps max`);
  if (state.minYear) filters.push(`From ${state.minYear}`);
  if (state.maxYear) filters.push(`Through ${state.maxYear}`);
  if (state.sortBy !== "relevance") {
    filters.push(els.sortSelect.selectedOptions[0]?.textContent || state.sortBy);
  }

  els.activeFilters.innerHTML = filters
    .map((filter) => `<span class="active-filter">${escapeHtml(filter)}</span>`)
    .join("");
  els.categoryCount.textContent = `${state.genres.size + state.excludedGenres.size} selected`;
  els.filterSummary.textContent = formatFilterSummary();
}

async function refreshRecommendations() {
  renderActiveFilters();
  renderPageBackdrop();
  updateResultsTitle();
  els.recommendationGrid.innerHTML = `<div class="loading">Scoring catalog matches...</div>`;

  const browsingFilters = hasActiveFilters();

  if (browsingFilters) {
    await loadCatalogPage(false);
    return;
  }

  const payload = await api("/api/recommend", {
    method: "POST",
    body: JSON.stringify({
      session_id: state.sessionId,
      // The user's likes are a *profile*, not a similarity constraint. Sending
      // them as reference_titles routed every personalized request to the
      // constraint-rich Hybrid, so the ALS + LambdaMART fast path never ran for
      // the main panel. liked_ids is what that path consumes.
      liked_ids: state.liked.map((item) => Number(item.id)).filter(Number.isFinite),
      include_genres: Array.from(state.genres),
      formats: state.mediaType ? [state.mediaType] : [],
      min_score: state.minScore ? Number(state.minScore) : null,
      max_episodes: state.maxEpisodes ? Number(state.maxEpisodes) : null,
      excluded_titles: state.preferences.excluded_titles,
      seen_titles: state.preferences.seen_titles,
      session_profile: preferencePayload(),
      // Liking one title should not return three entries from its franchise.
      // "More like this" already asked for this; the main panel now does too.
      one_per_series: true,
      limit: COMPACT_RESULT_LIMIT,
    }),
  });

  renderRecommendations(payload.results);
  els.loadMoreBtn.hidden = true;
  updateResultsTitle();
}

function catalogQueryPayload(offset) {
  return {
    query: state.searchQuery,
    include_genres: Array.from(state.genres),
    exclude_genres: Array.from(state.excludedGenres),
    formats: state.mediaType ? [state.mediaType] : [],
    required_studios: state.studio ? [state.studio] : [],
    min_score: state.minScore ? Number(state.minScore) : null,
    min_year: state.minYear ? Number(state.minYear) : null,
    max_year: state.maxYear ? Number(state.maxYear) : null,
    max_episodes: state.maxEpisodes ? Number(state.maxEpisodes) : null,
    sort_by: state.sortBy,
    semantic: Boolean(state.searchQuery),
    offset,
    top_k: CATALOG_PAGE_SIZE,
  };
}

async function loadCatalogPage(append) {
  const offset = append ? state.catalogOffset : 0;
  if (!append) {
    state.catalogOffset = 0;
    updateResultsTitle();
    els.recommendationGrid.innerHTML = `<div class="loading">Searching the catalog...</div>`;
  }
  els.loadMoreBtn.disabled = true;
  els.loadMoreBtn.textContent = "Loading";

  try {
    const payload = await api("/api/search", {
      method: "POST",
      body: JSON.stringify(catalogQueryPayload(offset)),
    });
    renderRecommendations(payload.results, append);
    state.catalogOffset = offset + payload.results.length;
    state.catalogTotal = payload.total;
    els.loadMoreBtn.dataset.offset = String(state.catalogOffset);
    els.loadMoreBtn.dataset.total = String(state.catalogTotal);
    els.loadMoreBtn.hidden = !payload.has_more;
    updateResultsTitle(payload.total);
  } finally {
    els.loadMoreBtn.disabled = false;
    els.loadMoreBtn.textContent = "Load more";
  }
}

function renderRecommendations(results, append = false) {
  if (!results.length && !append) {
    els.recommendationGrid.innerHTML = `<div class="loading">No recommendations match these filters.</div>`;
    return;
  }

  if (!append) state.currentResults = new Map();
  for (const item of results) state.currentResults.set(String(item.id), item);
  const cards = results.map(renderAnimeCard).join("");
  if (append) {
    els.recommendationGrid.insertAdjacentHTML("beforeend", cards);
  } else {
    els.recommendationGrid.innerHTML = cards;
  }
}

function renderAnimeCard(item) {
  const genres = (item.genres || []).slice(0, 3);
  const reasons = item.reasons || [];
  const cast = (item.characters || []).slice(0, 2).map((person) => person.name);
  const explanation = item.explanation_data || {};
  const strongest = explanation.strongest_channels || [];
  const matched = [
    ...(explanation.matched_genres || []),
    ...(explanation.matched_studios || []),
    ...(explanation.matched_creators || []),
  ];
  const scoreRows = Object.entries(item.score_breakdown?.channels || item.score_breakdown || {})
    .map(([key, value]) => `
      <span class="score-row">
        <span>${escapeHtml(key)}${value.active ? "" : " (inactive)"}</span>
        <strong>${Number(value.weighted_contribution || 0).toFixed(3)}</strong>
      </span>`)
    .join("");

  return `
    <article class="anime-card">
      <img src="${escapeHtml(item.image_url || "")}" alt="${escapeHtml(item.title)} poster" loading="lazy" />
      <div class="anime-body">
        <div>
          <h3 class="anime-title">${escapeHtml(item.title)}</h3>
          <div class="anime-meta">
            <span class="meta-pill">${escapeHtml(item.type || "Unknown")}</span>
            ${item.start_year ? `<span class="meta-pill">${escapeHtml(String(item.start_year))}</span>` : ""}
            <span class="meta-pill">Score ${escapeHtml(formatScore(item.score))}</span>
            ${item.episodes ? `<span class="meta-pill">${item.episodes} eps</span>` : ""}
          </div>
        </div>
        <div class="anime-meta">
          ${genres.map((genre) => `<span class="meta-pill">${escapeHtml(genre)}</span>`).join("")}
        </div>
        ${cast.length ? `<div class="people-line">${cast.map((name) => `<span class="meta-pill">${escapeHtml(name)}</span>`).join("")}</div>` : ""}
        <p class="synopsis">${escapeHtml(item.synopsis || "No synopsis available.")}</p>
        <ul class="reason-list">
          ${reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}
        </ul>
        <details class="why-panel">
          <summary>Why this recommendation?</summary>
          <div class="why-content">
            ${strongest.length ? `<p>Strongest channels: ${strongest.map(escapeHtml).join(", ")}</p>` : ""}
            ${matched.length ? `<p>Matched signals: ${matched.slice(0, 5).map(escapeHtml).join(", ")}</p>` : ""}
            ${(explanation.matched_constraints || []).length ? `<p>Constraints: ${explanation.matched_constraints.map(escapeHtml).join(", ")}</p>` : ""}
          </div>
        </details>
        ${
          // Channel weights only exist on the constraint-rich Hybrid path. The
          // fast ALS + LambdaMART path has no per-channel blend, and rendering
          // it there produced a panel of zeroes. Learned ranking scores are
          // deliberately not surfaced to readers either way.
          scoreRows
            ? `<details class="score-breakdown">
          <summary>Ranking signals</summary>
          <div class="score-grid">${scoreRows}</div>
        </details>`
            : ""
        }
        <div class="card-actions">
          <button class="secondary-button" type="button" data-detail-id="${item.id}">Details</button>
          <button class="icon-button" type="button" data-add-id="${item.id}" aria-label="Add ${escapeHtml(item.title)} to liked titles">+</button>
        </div>
        <div class="feedback-actions">
          <button class="secondary-button compact" type="button" data-feedback="like" data-id="${item.id}">Like</button>
          <button class="secondary-button compact" type="button" data-feedback="dislike" data-id="${item.id}">Dislike</button>
          <button class="secondary-button compact" type="button" data-feedback="watched" data-id="${item.id}">Watched</button>
          <button class="secondary-button compact" type="button" data-feedback="exclude" data-id="${item.id}">Exclude</button>
          <button class="secondary-button compact" type="button" data-more-like-id="${item.id}">More like this</button>
        </div>
      </div>
    </article>
  `;
}

function addMessage(role, content) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  const paragraph = document.createElement("p");
  paragraph.textContent = content;
  node.appendChild(paragraph);
  els.chatLog.appendChild(node);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function addTypingIndicator() {
  const node = document.createElement("div");
  node.className = "message assistant typing";
  node.setAttribute("aria-label", "Compass is thinking");
  node.innerHTML = '<span class="progress-label">Understanding your request</span><span></span><span></span><span></span>';
  els.chatLog.appendChild(node);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
  return node;
}

function resizeAgentInput() {
  els.agentInput.style.height = "auto";
  els.agentInput.style.height = `${Math.min(els.agentInput.scrollHeight, 160)}px`;
}

async function sendAgentMessage(message) {
  const requestHistory = state.history.slice(-8);
  addMessage("user", message);
  state.history.push({ role: "user", content: message });
  els.agentStarters.hidden = true;
  els.agentInput.value = "";
  resizeAgentInput();
  els.agentDock.setAttribute("aria-busy", "true");
  const typingIndicator = addTypingIndicator();
  const progressStages = [
    "Understanding your request",
    "Resolving catalog entities",
    "Applying filters",
    "Ranking candidates",
    "Generating explanations",
  ];
  let progressIndex = 0;
  const progressTimer = window.setInterval(() => {
    progressIndex = Math.min(progressIndex + 1, progressStages.length - 1);
    const label = typingIndicator.querySelector(".progress-label");
    if (label) label.textContent = progressStages[progressIndex];
  }, 850);

  try {
    const payload = await api("/api/agent", {
      method: "POST",
      body: JSON.stringify({ message, history: requestHistory, session_id: state.sessionId }),
    });

    updateAgentStatus(payload.agent);
    addMessage("assistant", payload.answer);
    state.history.push({ role: "assistant", content: payload.answer });
    state.lastTrace = payload.trace || [];
    els.traceOutput.textContent = state.lastTrace.length ? JSON.stringify(state.lastTrace, null, 2) : "No tool calls for this response.";
  } finally {
    window.clearInterval(progressTimer);
    typingIndicator.remove();
    els.agentDock.setAttribute("aria-busy", "false");
  }
}

async function applyFeedback(action, id) {
  const item = state.currentResults.get(String(id)) || (await api(`/api/anime/${id}`)).result;

  if (action === "like") {
    addLiked(item);
    return;
  }

  if (action === "dislike") {
    rememberPreference("disliked_titles", item.title);
    rememberPreference("excluded_titles", item.title);
  }

  if (action === "watched") {
    rememberPreference("seen_titles", item.title);
  }

  if (action === "exclude") {
    rememberPreference("excluded_titles", item.title);
  }

  await syncSessionProfile();
  await refreshRecommendations();
}

async function showMoreLike(id) {
  const item = state.currentResults.get(String(id)) || (await api(`/api/anime/${id}`)).result;
  state.preferences.previous_reference_titles = [item.title];
  saveSessionState();
  els.resultsTitle.textContent = `More like ${item.title}`;
  els.recommendationGrid.innerHTML = `<div class="loading">Finding similar titles...</div>`;

  const payload = await api("/api/recommend", {
    method: "POST",
    body: JSON.stringify({
      session_id: state.sessionId,
      reference_titles: [item.title],
      include_genres: Array.from(state.genres),
      formats: state.mediaType ? [state.mediaType] : [],
      min_score: state.minScore,
      max_episodes: state.maxEpisodes,
      excluded_titles: [item.title, ...state.preferences.excluded_titles],
      seen_titles: state.preferences.seen_titles,
      session_profile: preferencePayload(),
      one_per_series: true,
      top_k: COMPACT_RESULT_LIMIT,
    }),
  });

  renderRecommendations(payload.results);
}

async function openDetails(id) {
  const payload = await api(`/api/anime/${id}`);
  renderDetails(payload.result);
  els.detailModal.hidden = false;
}

function closeDetails() {
  els.detailModal.hidden = true;
}

function renderDetails(item) {
  const characters = item.characters || [];
  const staff = item.staff || [];
  const voiceActors = item.voice_actors || [];

  els.detailContent.innerHTML = `
    <section class="detail-hero">
      <img class="detail-poster" src="${escapeHtml(item.image_url || "")}" alt="${escapeHtml(item.title)} poster" />
      <div>
        <p class="eyebrow">${escapeHtml(formatMeta(item))}</p>
        <h2 id="detailTitle" class="detail-title">${escapeHtml(item.title)}</h2>
        <div class="anime-meta">
          ${(item.genres || []).map((genre) => `<span class="meta-pill">${escapeHtml(genre)}</span>`).join("")}
        </div>
        <p class="synopsis">${escapeHtml(item.synopsis || "No synopsis available.")}</p>
        ${renderLabelList("Studios", item.studios)}
        ${renderLabelList("Producers", item.producers)}
      </div>
    </section>
    ${renderPeopleSection("Characters and Voice Cast", characters, "character")}
    ${renderPeopleSection("Voice Actors", voiceActors, "voice")}
    ${renderPeopleSection("Staff", staff, "staff")}
  `;
}

function renderLabelList(title, values = []) {
  if (!values.length) return "";
  return `
    <div class="detail-section" style="padding: 16px 0 0;">
      <h3>${escapeHtml(title)}</h3>
      <div class="anime-meta">${values.map((value) => `<span class="meta-pill">${escapeHtml(value)}</span>`).join("")}</div>
    </div>
  `;
}

function renderPeopleSection(title, people, mode) {
  if (!people.length) return "";
  return `
    <section class="detail-section">
      <h3>${escapeHtml(title)}</h3>
      <div class="people-grid">
        ${people.map((person) => renderPerson(person, mode)).join("")}
      </div>
    </section>
  `;
}

function renderPerson(person, mode) {
  const subtitle = mode === "character"
    ? `${person.role || "Character"}${person.voice_actors?.length ? ` / VA: ${person.voice_actors[0].name}` : ""}`
    : `${person.role || person.language || "Contributor"}`;

  return `
    <article class="person-card">
      <img src="${escapeHtml(person.image_url || "")}" alt="" loading="lazy" />
      <div>
        <p class="item-title">${escapeHtml(person.name)}</p>
        <p class="item-meta">${escapeHtml(subtitle)}</p>
      </div>
    </article>
  `;
}

els.searchInput.addEventListener(
  "input",
  debounce((event) => {
    searchAnime(event.target.value.trim(), false).catch((error) => {
      els.searchResults.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    });
  }),
);

els.searchInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  submitSearch(els.searchInput.value).catch((error) => {
    els.recommendationGrid.innerHTML = `<div class="loading">${escapeHtml(error.message)}</div>`;
  });
});

els.searchInput.addEventListener("focus", () => {
  searchAnime(els.searchInput.value.trim(), true).catch((error) => {
    els.searchResults.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  });
});

document.addEventListener("click", async (event) => {
  const clickedInsideFilter = Boolean(event.target.closest(".filter-shell"));
  const clickedInsideAgent = Boolean(event.target.closest(".agent-dock"));
  const clickedAgentToggle = Boolean(event.target.closest("#agentToggle"));
  const addButton = event.target.closest("[data-add-id]");
  const detailButton = event.target.closest("[data-detail-id]");
  const feedbackButton = event.target.closest("[data-feedback]");
  const moreLikeButton = event.target.closest("[data-more-like-id]");
  const removeButton = event.target.closest("[data-remove-id]");
  const closeButton = event.target.closest("[data-close-modal]");
  const closeAgentButton = event.target.closest("#closeAgentBtn");
  const closeFiltersButton = event.target.closest("#closeFiltersBtn");
  const loadMoreButton = event.target.closest("#loadMoreBtn");

  if (loadMoreButton) {
    try {
      await loadCatalogPage(true);
    } catch (error) {
      els.loadMoreBtn.textContent = error.message;
    }
    return;
  }

  if (feedbackButton) {
    await applyFeedback(feedbackButton.dataset.feedback, feedbackButton.dataset.id);
    return;
  }

  if (moreLikeButton) {
    await showMoreLike(moreLikeButton.dataset.moreLikeId);
    return;
  }

  if (addButton) {
    const id = Number(addButton.dataset.addId);
    const result = await api(`/api/anime/${id}`);
    addLiked(result.result);
    return;
  }

  if (detailButton) {
    await openDetails(Number(detailButton.dataset.detailId));
    return;
  }

  if (removeButton) {
    removeLiked(removeButton.dataset.removeId);
    return;
  }

  if (closeButton) {
    closeDetails();
    return;
  }

  if (closeAgentButton) {
    setAgentOpen(false);
    return;
  }

  if (closeFiltersButton) {
    setFilterOpen(false);
    return;
  }

  if (!event.target.closest(".top-search")) {
    searchRequestSequence += 1;
    els.searchResults.innerHTML = "";
  }

  if (state.filterOpen && !clickedInsideFilter) {
    setFilterOpen(false);
  }

  if (state.agentOpen && !clickedInsideAgent && !clickedAgentToggle) {
    setAgentOpen(false);
  }
});

els.clearTasteBtn.addEventListener("click", () => {
  state.liked = [];
  saveSessionState();
  syncSessionProfile().catch(console.error);
  renderLiked();
  refreshRecommendations();
});

els.resetSessionBtn.addEventListener("click", () => {
  const oldSessionId = state.sessionId;
  state.sessionId = createSessionId();
  state.liked = [];
  state.preferences = defaultPreferences();
  saveSessionState();
  api("/api/session/preferences", {
    method: "POST",
    body: JSON.stringify({ session_id: oldSessionId, reset: true }),
  }).catch(console.error);
  renderLiked();
  refreshRecommendations();
});

els.clearGenresBtn.addEventListener("click", () => {
  state.genres.clear();
  state.excludedGenres.clear();
  document.querySelectorAll(".genre-chip.active").forEach((chip) => chip.classList.remove("active"));
  renderGenreChips(state.allGenres);
  refreshRecommendations();
});

els.filterToggle.addEventListener("click", () => {
  setFilterOpen(!state.filterOpen);
});

els.agentToggle.addEventListener("click", () => {
  setAgentOpen(true);
});

els.agentStarters.addEventListener("click", (event) => {
  const button = event.target.closest("[data-agent-prompt]");
  if (!button) return;
  els.agentInput.value = button.dataset.agentPrompt;
  resizeAgentInput();
  els.agentForm.requestSubmit();
});

els.genreChips.addEventListener("click", (event) => {
  const button = event.target.closest("[data-genre]");
  if (!button) return;

  const genre = button.dataset.genre;
  if (state.genres.has(genre)) {
    state.genres.delete(genre);
    state.excludedGenres.add(genre);
  } else if (state.excludedGenres.has(genre)) {
    state.excludedGenres.delete(genre);
  } else {
    state.genres.add(genre);
  }
  renderGenreChips(state.allGenres);
  refreshRecommendations();
});

els.typeSelect.addEventListener("change", (event) => {
  state.mediaType = event.target.value;
  els.filterSummary.textContent = formatFilterSummary();
  refreshRecommendations();
});

els.studioInput.addEventListener("change", (event) => {
  state.studio = event.target.value.trim();
  refreshRecommendations();
});

els.minScoreInput.addEventListener("change", (event) => {
  state.minScore = event.target.value;
  refreshRecommendations();
});

els.maxEpisodesInput.addEventListener("change", (event) => {
  state.maxEpisodes = event.target.value;
  refreshRecommendations();
});

els.minYearInput.addEventListener("change", (event) => {
  state.minYear = event.target.value;
  refreshRecommendations();
});

els.maxYearInput.addEventListener("change", (event) => {
  state.maxYear = event.target.value;
  refreshRecommendations();
});

els.sortSelect.addEventListener("change", (event) => {
  state.sortBy = event.target.value;
  refreshRecommendations();
});

els.topRecommendBtn.addEventListener("click", () => {
  const query = els.searchInput.value.trim();
  if (query) {
    submitSearch(query).catch((error) => {
      els.recommendationGrid.innerHTML = `<div class="loading">${escapeHtml(error.message)}</div>`;
    });
    return;
  }
  refreshRecommendations();
});

els.agentForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = els.agentInput.value.trim();
  if (!message) return;

  const button = els.agentForm.querySelector("button");
  button.disabled = true;
  button.textContent = "Thinking";
  sendAgentMessage(message)
    .catch((error) => addMessage("assistant", error.message))
    .finally(() => {
      button.disabled = false;
      button.textContent = "Send";
    });
});

els.agentInput.addEventListener("input", resizeAgentInput);

els.agentInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.agentForm.requestSubmit();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.detailModal.hidden) {
    closeDetails();
    return;
  }

  if (event.key === "Escape") {
    setFilterOpen(false);
    setAgentOpen(false);
  }
});

initialize();
