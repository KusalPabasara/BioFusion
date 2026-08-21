/**
 * BioFusion Kiosk — Client-side JavaScript
 * Handles HTMX events and camera preview lifecycle.
 */

// ─── HTMX Event Hooks ──────────────────────────────────────────────────────

// After any HTMX swap, handle cleanup
document.addEventListener('htmx:afterSwap', function(event) {
    // If we swapped back to idle, ensure camera stream is stopped
    const preview = document.getElementById('camera-preview');
    if (!preview) {
        // No preview on screen — camera stream will stop naturally
        // since the <img> tag requesting /api/preview is removed
    }
});

// Before a request, disable the triggering button to prevent double-clicks
document.addEventListener('htmx:beforeRequest', function(event) {
    const trigger = event.detail.elt;
    if (trigger && trigger.tagName === 'BUTTON') {
        trigger.disabled = true;
        trigger.style.opacity = '0.6';
    }
});

// After request completes (success or error), re-enable buttons
document.addEventListener('htmx:afterRequest', function(event) {
    const trigger = event.detail.elt;
    if (trigger && trigger.tagName === 'BUTTON') {
        trigger.disabled = false;
        trigger.style.opacity = '1';
    }
});

// Handle request errors gracefully
document.addEventListener('htmx:responseError', function(event) {
    console.error('HTMX request error:', event.detail);
});

// ─── Prevent zoom on double-tap (kiosk mode) ───────────────────────────────
document.addEventListener('dblclick', function(e) {
    e.preventDefault();
}, { passive: false });

// ─── Prevent context menu (kiosk mode) ──────────────────────────────────────
document.addEventListener('contextmenu', function(e) {
    e.preventDefault();
});

// ─── Console greeting ───────────────────────────────────────────────────────
console.log('%c🏥 BioFusion Kiosk', 'font-size: 20px; font-weight: bold; color: #2563EB;');
console.log('%cAI-Assisted Pneumonia Detection System', 'font-size: 12px; color: #94A3B8;');
