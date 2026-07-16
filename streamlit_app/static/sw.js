/* BioFusion service worker.
 *
 * Goal: make the app installable and give it a graceful offline shell — WITHOUT
 * caching Streamlit's dynamic traffic (its WebSocket stream and internal APIs),
 * which must always hit the network for live inference to work.
 */
const CACHE = "biofusion-v1";
const SHELL = [
  "/static/manifest.webmanifest",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/offline.html",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never intercept Streamlit's live endpoints or non-GET/cross-origin requests.
  if (
    event.request.method !== "GET" ||
    url.origin !== self.location.origin ||
    url.pathname.startsWith("/_stcore") ||
    url.pathname.startsWith("/_stcore/stream") ||
    url.pathname.startsWith("/component") ||
    url.pathname.startsWith("/media") ||
    url.pathname.startsWith("/upload")
  ) {
    return; // let the browser handle it normally
  }

  // Static assets: cache-first (they're versioned/stable).
  if (url.pathname.startsWith("/static/")) {
    event.respondWith(
      caches.match(event.request).then((hit) =>
        hit ||
        fetch(event.request).then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(event.request, copy));
          return res;
        })
      )
    );
    return;
  }

  // Navigations: network-first, falling back to the offline shell when down.
  if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request).catch(() => caches.match("/static/offline.html"))
    );
  }
});
