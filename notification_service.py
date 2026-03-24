from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Optional

from dotenv import load_dotenv

# Charger les variables depuis .env
load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Variable d'environnement manquante : {name}")
    return value


def _build_message(
    subject: str,
    body: str,
    to_email: str,
    attachment_bytes: bytes | None = None,
    filename: str = "report.pdf",
    attachment_mime_main: str = "application",
    attachment_mime_sub: str = "pdf",
) -> EmailMessage:
    smtp_user = _get_env("SMTP_USER")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg.set_content(body)

    if attachment_bytes:
        msg.add_attachment(
            attachment_bytes,
            maintype=attachment_mime_main,
            subtype=attachment_mime_sub,
            filename=filename,
        )

    return msg


def send_email(
    subject: str,
    body: str,
    to_email: str,
    attachment_bytes: bytes | None = None,
    filename: str = "report.pdf",
) -> None:
    """
    Envoie un email simple ou avec PDF en pièce jointe.

    Variables d'environnement attendues :
    - SMTP_HOST
    - SMTP_PORT
    - SMTP_USER
    - SMTP_PASSWORD
    - SMTP_FROM (optionnel)
    """
    smtp_host = _get_env("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(_get_env("SMTP_PORT", "465"))
    smtp_user = _get_env("SMTP_USER")
    smtp_password = _get_env("SMTP_PASSWORD")

    msg = _build_message(
        subject=subject,
        body=body,
        to_email=to_email,
        attachment_bytes=attachment_bytes,
        filename=filename,
    )

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)


# =========================================================
# FONCTIONS MÉTIER LIÉES AUX EXPERTS 1→4
# =========================================================
def send_station_report_email(
    station_id: str,
    to_email: str,
    pdf_bytes: bytes,
    custom_message: str | None = None,
) -> None:
    """
    Envoie un rapport station PDF.
    Compatible Expert 1, 2, 3, 4.
    """
    subject = f"Rapport CleanWater — Station {station_id}"

    body = custom_message or (
        f"Bonjour,\n\n"
        f"Veuillez trouver en pièce jointe le rapport CleanWater de la station {station_id}.\n\n"
        f"Ce rapport contient :\n"
        f"- les mesures capteurs\n"
        f"- le scoring qualité / risque\n"
        f"- le diagnostic automatique\n"
        f"- les graphiques temporels\n"
        f"- l’analyse satellite Sentinel\n"
        f"- les recommandations d’aide à la décision\n\n"
        f"Cordialement."
    )

    send_email(
        subject=subject,
        body=body,
        to_email=to_email,
        attachment_bytes=pdf_bytes,
        filename=f"station_report_{station_id}.pdf",
    )


def send_global_report_email(
    to_email: str,
    pdf_bytes: bytes,
    custom_message: str | None = None,
) -> None:
    """
    Envoie un rapport global PDF.
    Compatible Expert 1, 2, 3, 4.
    """
    subject = "Rapport Global CleanWater"

    body = custom_message or (
        "Bonjour,\n\n"
        "Veuillez trouver en pièce jointe le rapport global CleanWater.\n\n"
        "Ce rapport contient :\n"
        "- la synthèse des stations\n"
        "- les scores globaux\n"
        "- les diagnostics détaillés\n"
        "- les recommandations prioritaires\n\n"
        "Cordialement."
    )

    send_email(
        subject=subject,
        body=body,
        to_email=to_email,
        attachment_bytes=pdf_bytes,
        filename="global_report_cleanwater.pdf",
    )


def send_alert_notification_email(
    station_id: str,
    to_email: str,
    alert_title: str,
    alert_message: str,
    pdf_bytes: bytes | None = None,
) -> None:
    """
    Envoie une alerte email avec pièce jointe optionnelle.
    Très utile pour Expert 4.
    """
    subject = f"🚨 Alerte CleanWater — {station_id} — {alert_title}"

    body = (
        f"Bonjour,\n\n"
        f"Une alerte a été déclenchée sur la station {station_id}.\n\n"
        f"Titre : {alert_title}\n"
        f"Détail : {alert_message}\n\n"
        f"Merci de consulter la situation rapidement."
    )

    send_email(
        subject=subject,
        body=body,
        to_email=to_email,
        attachment_bytes=pdf_bytes,
        filename=f"alert_report_{station_id}.pdf" if pdf_bytes else "alert.pdf",
    )


# =========================================================
# TEST SIMPLE
# =========================================================
if __name__ == "__main__":
    # Test minimal sans pièce jointe
    try:
        test_to = _get_env("SMTP_USER")
        send_email(
            subject="Test CleanWater",
            body="Ceci est un email de test envoyé depuis notification_service.py",
            to_email=test_to,
        )
        print("Email de test envoyé avec succès.")
    except Exception as e:
        print("Erreur lors du test email :", e)