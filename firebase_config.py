import os
import firebase_admin
from firebase_admin import credentials, firestore

_db = None


def init_firebase():
    global _db

    if _db is not None:
        return _db

    key_path = os.getenv("FIREBASE_KEY_PATH", "serviceAccountKey.json")

    if not os.path.exists(key_path):
        raise FileNotFoundError(
            f"Clé Firebase introuvable : {key_path}. "
            "Place serviceAccountKey.json à la racine du projet "
            "ou définis FIREBASE_KEY_PATH."
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)

    _db = firestore.client()
    return _db