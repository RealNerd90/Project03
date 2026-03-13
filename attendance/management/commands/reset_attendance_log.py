from django.core.management.base import BaseCommand

from attendance.models import AttendanceRecord


class Command(BaseCommand):
    help = "Delete all AttendanceRecord rows (reset attendance log)."

    def handle(self, *args, **options):
        count = AttendanceRecord.objects.count()
        AttendanceRecord.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f"Attendance log reset: deleted {count} record(s)."))

