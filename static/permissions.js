/**
 * Permissions Utility for AttendancePro
 * Handles Camera, Location, and Notification permissions with user-triggered prompts.
 */

const PermissionsManager = {
    isSecureContext: window.isSecureContext,

    /**
     * Checks if the current context is secure (HTTPS or localhost)
     * @returns {boolean}
     */
    checkSecureContext: function() {
        if (!this.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
            return false;
        }
        return true;
    },

    /**
     * Requests Camera permission
     * @returns {Promise<MediaStream|null>}
     */
    requestCamera: async function() {
        try {
            return await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            });
        } catch (err) {
            console.error('Camera access denied:', err);
            return null;
        }
    },

    /**
     * Requests Location permission
     * @returns {Promise<{lat: number, lon: number}|null>}
     */
    requestLocation: function() {
        return new Promise((resolve) => {
            if (!navigator.geolocation) {
                resolve(null);
                return;
            }
            navigator.geolocation.getCurrentPosition(
                (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
                (err) => {
                    console.error('Location access denied:', err);
                    resolve(null);
                },
                { enableHighAccuracy: true, timeout: 5000 }
            );
        });
    },

    /**
     * Requests Notification permission
     * @returns {Promise<string>}
     */
    requestNotifications: async function() {
        if (!('Notification' in window)) return 'unsupported';
        try {
            return await Notification.requestPermission();
        } catch (err) {
            console.error('Notification request failed:', err);
            return 'denied';
        }
    },

    /**
     * Unified function to request all necessary permissions at once
     * Triggered by a single user gesture (button click)
     */
    requestAll: async function() {
        const results = {
            camera: null,
            location: null,
            notifications: null
        };

        // 1. Notifications
        results.notifications = await this.requestNotifications();

        // 2. Camera (returns stream if granted)
        results.camera = await this.requestCamera();
        if (results.camera) {
            // Immediately stop the stream as we only wanted to prompt
            results.camera.getTracks().forEach(track => track.stop());
            results.camera = 'granted';
        } else {
            results.camera = 'denied';
        }

        // 3. Location
        const loc = await this.requestLocation();
        results.location = loc ? 'granted' : 'denied';

        return results;
    }
};

window.PermissionsManager = PermissionsManager;
