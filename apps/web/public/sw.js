const STATIC_CACHE = "qpk-static-v1";

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(STATIC_CACHE).then((cache) => cache.addAll(["/manifest.webmanifest", "/qpk-mark.svg"])));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== STATIC_CACHE).map((key) => caches.delete(key)))),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/") || event.request.mode === "navigate") return;
  if (!url.pathname.startsWith("/_next/static/") && !url.pathname.endsWith(".svg")) return;
  event.respondWith(
    caches.match(event.request).then(
      (cached) =>
        cached ||
        fetch(event.request).then((response) => {
          const clone = response.clone();
          void caches.open(STATIC_CACHE).then((cache) => cache.put(event.request, clone));
          return response;
        }),
    ),
  );
});
