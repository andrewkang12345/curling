/* Curling Arena service worker: cache the app shell; API always from network. */
const VERSION = "arena-v2.4.0";
const SHELL = ["/", "/static/style_v2.css", "/static/app_v2.js",
               "/static/icons/icon-192.png", "/manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))
  ).then(() => self.clients.claim()));
});
const CODE = ["/", "/static/app_v2.js", "/static/style_v2.css"];

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/")) return;  // network
  const isCode = CODE.includes(url.pathname);
  e.respondWith(
    caches.match(e.request).then((hit) => {
      const fetched = fetch(e.request).then((resp) => {
        if (resp.ok) {
          const clone = resp.clone();
          caches.open(VERSION).then((c) => c.put(e.request, clone));
        }
        return resp;
      }).catch(() => hit);
      // code shell: network-first (stale cached clients caused desktop bugs);
      // icons/assets: cache-first
      return isCode ? fetched : (hit || fetched);
    })
  );
});
