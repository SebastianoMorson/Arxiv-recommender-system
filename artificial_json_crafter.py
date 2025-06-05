import json
from datetime import datetime
import random


def craft_uactions_from_samples():
    # Carica i dati dal file registration_samples.json
    with open('registration_samples.json', 'r') as f:
        samples = json.load(f)

    user_actions = {}
    now = datetime.now().isoformat()

    for entry in samples:
        # Prendi l'id se presente
        paper_id = entry.get("id")
        if not paper_id:
            continue  # salta se manca l'id

        # Valutazione artificiale: clicks sempre "1", likes random tra -1, 0, 1
        user_actions[paper_id] = {
            "clicks": ["1"],
            "likes": [str(random.choice([-1, 0, 1]))],
            "favorites": ["1"],
            "time_spent": [],
            "searches": [],
            "last_active": now
        }

    # Salva il risultato su user_actions.json
    with open('user_actions.json', 'w') as f:
        json.dump(user_actions, f, indent=4)


def craft_uactions_from_userrec():
    # Carica i dati dal file userrec.json
    with open('userrec.json', 'r') as f:
        userrec = json.load(f)

    user_actions = {}
    now = datetime.now().isoformat()

    for user_id, recs in userrec.items():
        for rec in recs:
            paper_id = rec.get("id")
            if not paper_id:
                continue  # salta se manca l'id

            # Valutazione artificiale: clicks sempre "1", likes random tra -1, 0, 1
            user_actions[paper_id] = {
                "clicks": ["1"],
                "likes": [str(random.choice([-1, 0, 1]))],
                "favorites": ["1"],
                "time_spent": [],
                "searches": [],
                "last_active": now
            }

    # Salva il risultato su user_actions.json
    with open('user_actions.json', 'w') as f:
        json.dump(user_actions, f, indent=4)



if __name__ == "__main__":
    # in base all'argomento passato a --from decido quale funzione chiamare
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--from":
        if len(sys.argv) > 2 and sys.argv[2] == "samples":
            craft_uactions_from_samples()
        elif len(sys.argv) > 2 and sys.argv[2] == "userrec":
            craft_uactions_from_userrec()
        else:
            print("Usage: python artificial_json_crafter.py --from [samples|userrec]")
    else:
        print("Usage: python artificial_json_crafter.py --from [samples|userrec]")
# Esempio di utilizzo:
# python artificial_json_crafter.py --from samples
# python artificial_json_crafter.py --from userrec
# Questo script crea un file user_actions.json a partire da registration_samples.json o userrec.json