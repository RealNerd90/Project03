from django.db import models


class RegisteredUser(models.Model):
    """Basic registry of people known to the face-recognition system."""

    name = models.CharField(max_length=255, unique=True)
    email = models.EmailField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name} <{self.email or 'no-email'}>"


class AttendanceRecord(models.Model):
    """Single attendance event produced by the face-recognition system."""

    STATUS_PRESENT = "present"
    STATUS_OUT_OF_RADIUS = "out_of_radius"

    STATUS_CHOICES = [
        (STATUS_PRESENT, "Present"),
        (STATUS_OUT_OF_RADIUS, "Out of radius"),
    ]

    name = models.CharField(max_length=255)
    date = models.DateField()
    time = models.TimeField(null=True, blank=True)
    status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default=STATUS_PRESENT,
    )
    check_out_time = models.TimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date", "-time", "-created_at"]

    def __str__(self) -> str:
        return f"{self.name} - {self.date} {self.time or ''} ({self.status})"

