# Smart Attendance System: Presentation Guide

This guide contains everything you need for your presentation on Monday.

---

## 1. The "Big Idea" (The 30-Second Pitch)
"This is a **Smart Attendance System** that uses **AI-powered facial recognition** and **Geofencing** to automate attendance. It ensures that users are physically present at the correct location and securely identifies them without any manual paperwork."

---

## 2. The AI Facial Recognition (How it "Sees")
*   **Detection (MTCNN):** This is the "Detector." It scans the camera feed, identifies eyes, nose, and mouth, and draws a box around the face. This removes background noise.
*   **Recognition (FaceNet):** This is the "Translator." It takes the face image and creates an **Embedding**—a unique "Face Fingerprint" made of 128 numbers.
*   **Matching:** The system compares the current fingerprint with the one stored in the database. A match verifies the user’s identity.

---

## 3. The Geofencing Logic (The Security Boundary)
*   **The Problem:** Preventing "Check-ins from home."
*   **The Solution:** Using the browser's **HTML5 Geolocation API** and the **Haversine Formula**.
*   **The Logic:** Django calculates the mathematical distance between the user’s GPS coordinates and the "Office Center." If the distance is less than the allowed **Radius** (e.g., 500m), the attendance is marked valid.

---

## 4. The Architecture (The Django "Chef" Analogy)
We use the **MVT (Model-View-Template)** architecture:
*   **Model (The Pantry):** Defines our database tables (Users, Records, Sounds).
*   **Template (The Plate):** The HTML and CSS that the user interacts with.
*   **View (The Chef):** The logic in `views.py` that takes the user's request, processes the AI/Geofencing logic, and returns the result.

---

## 5. Key Technical Features
*   **Real-Time Polling:** The system checks for scheduled reminders every 30 seconds using JavaScript AJAX, so the page never needs a manual refresh.
*   **Dynamic Sound Library:** Admins can upload custom MP3/WAV files, which are validated (max 2MB) and stored securely in the database.
*   **Secure Authentication:** Passwords are never stored as text; they are protected by AES-encrypted hashing.

---

## 6. Database Architecture (The "Memory" of the System)
The system uses an **SQLite3** relational database. Below are the key tables (Models) and their purposes:

### A. `RegisteredUser`
*   **Purpose:** Stores the core profile of every authorized person.
*   **Key Fields:** Name, Email, Encrypted Password, Phone, Gender, and Account Role (Student/Employee/Teacher).
*   **Reason:** This is our "Identity Registry" used to verify who is standing in front of the camera.

### B. `AttendanceRecord`
*   **Purpose:** The central log for all check-in and check-out events.
*   **Key Fields:** User Name, Date, Check-in Time, Check-out Time, Status (Present/Out of Radius), and GPS Coordinates (Lat/Lon).
*   **Reason:** Provides the history and reporting data for HR/Management to see who was present and when.

### C. `AdminAccount`
*   **Purpose:** Secure storage for administrator credentials.
*   **Key Fields:** Admin Email and secure Password Hash.
*   **Reason:** Ensures only authorized managers can access the Geofencing and System settings.

### D. `GeofenceSetting`
*   **Purpose:** Stores the physical boundaries of the office/campus.
*   **Key Fields:** Site Name, Latitude, Longitude, and allowed Radius (in meters).
*   **Reason:** This is the "Anchor Point" used by the Haversine formula to verify user location.

### E. `SystemSetting`
*   **Purpose:** Global configuration for the platform's behavior.
*   **Key Fields:** Session Timeout, Max Login Attempts, Maintenance Mode toggle, and the Attendance Reminder (Time and Sound toggle).
*   **Reason:** Allows the Admin to customize the system experience without touching a single line of code.

### F. `ReminderSound`
*   **Purpose:** Manages the custom notification sounds library.
*   **Key Fields:** Sound Name and the Audio File path.
*   **Reason:** Stores the metadata for custom-uploaded sounds so they can be played back during reminders.

---

## 7. Closing Statement
> *"In conclusion, this system transforms attendance from a manual, vulnerable process into a secure, AI-driven ecosystem that ensures both physical presence and verified identity."*

---

## 8. Pro-Tips for Demo
1.  **Show the Admin Dashboard:** It shows the power of the control center.
2.  **Upload a Sound:** Shows the library is functional and validated.
3.  **Explain the "Why":** Focus on security, efficiency, and removing human error.
