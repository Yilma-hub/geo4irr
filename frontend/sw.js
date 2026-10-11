/* Geo4Irr service worker.
 *
 * What this is for, and what it is deliberately not for.
 *
 * It caches the app itself — the one HTML file and the statewide field layer —
 * so the tool opens instantly and still opens when the signal drops on a farm
 * road. That is the whole job.
 *
 * It does NOT cache live data. Satellite tiles, weather, the soil survey and
 * the API all go to the network every time, because a cached irrigation
 * decision is a wrong irrigation decision. A grower looking at yesterday's NDVI
 * believing it is today's is worse off than a grower who can see the request
 * failed. So: cache the shell, never the answers.
 */

const VERSION = "geo4irr-v1";
const SHELL = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icon.svg",
  "./data/irrigated_areas.js",
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(VERSION)
      // addAll is all-or-nothing: one missing file and nothing is cached at
      // all. Each is added on its own so a stray 404 cannot leave the app
      // uninstallable.
      .then(cache => Promise.all(
        SHELL.map(url => cache.add(url).catch(err =>
          console.warn("[geo4irr] could not cache", url, err)))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== VERSION).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

/* Hosts whose answers must never be served from a cache. */
const LIVE_HOSTS = [
  "titiler.xyz",
  "earth-search.aws.element84.com",
  "sdmdataaccess.sc.egov.usda.gov",
  "api.weather.gov",
  "deos.udel.edu",
  "services.deos.udel.edu",
];

function isLiveData(url) {
  if (LIVE_HOSTS.some(h => url.hostname.endsWith(h))) return true;
  if (url.pathname.startsWith("/api/")) return true;
  return false;
}

self.addEventListener("fetch", event => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  // Live data and anything cross-origin that is not the map basemap: straight
  // to the network, no cache, no fallback. Failing loudly is correct here.
  if (isLiveData(url)) return;

  // Map tiles and the Leaflet library come from CDNs. Let the browser's own
  // HTTP cache handle those; a service worker cache on top only adds a way for
  // them to go stale.
  if (url.origin !== self.location.origin) return;

  // The app shell: serve from cache immediately, then refresh it in the
  // background so the next launch has the newer copy.
  event.respondWith(
    caches.match(req).then(cached => {
      const network = fetch(req).then(res => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(VERSION).then(c => c.put(req, copy));
        }
        return res;
      }).catch(() => cached);

      return cached || network;
    })
  );
});
