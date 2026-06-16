const CACHE = 'cantos-da-mata-v1';

const ASSETS = [
    '/',
    '/static/dia-da-mata-atlantica.jpg',
    '/static/Araponga.jpg',
    '/static/BemTeVi.jpg',
    '/static/Urutau.jpg',
    '/static/JoaoDeBarro.jpg',
    '/static/Tucano.jpg',
    '/static/Walking_Creature.json',
];

self.addEventListener('install', e => {
    e.waitUntil(
        caches.open(CACHE).then(cache => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', e => {
    e.respondWith(
        caches.match(e.request).then(cached => cached || fetch(e.request))
    );
});