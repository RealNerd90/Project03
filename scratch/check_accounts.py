import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from attendance.models import RegisteredUser, AdminAccount

def check_accounts():
    print("--- AdminAccount List ---")
    admins = AdminAccount.objects.all()
    for a in admins:
        print(f"ID: {a.id}, Email: {a.email}")

    print("\n--- RegisteredUser List ---")
    users = RegisteredUser.objects.all()
    for u in users:
        print(f"ID: {u.id}, Name: {u.name}, Email: {u.email}")

if __name__ == "__main__":
    check_accounts()
