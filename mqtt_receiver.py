from __future__ import annotations

import json
import paho.mqtt.client as mqtt

from workers.ingestion_worker import process_measurement

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "cleanwater/+/data"


def on_connect(client, userdata, flags, rc):
    print("Connecté au broker MQTT, code =", rc)
    client.subscribe(TOPIC)
    print("Abonné au topic :", TOPIC)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        result = process_measurement(payload)
        print("Mesure traitée :", result["status"], "| alertes =", result["alerts_count"])
    except Exception as e:
        print("Erreur MQTT :", e)


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()