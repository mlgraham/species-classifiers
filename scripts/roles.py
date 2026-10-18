"""Hive roles: one table that maps a species to what it means at a hive entrance.

The hive models keep a species-level head; device code and evaluation sum the
species probabilities into these roles and act on the roles. Training,
scripts/eval_roles.py and roles.json all use this table, so the roll-up is the
same everywhere.
"""
import json

BEE_FAMILIES = {"Apidae", "Megachilidae", "Halictidae", "Andrenidae", "Colletidae", "Melittidae", "Stenotritidae"}

# role -> (kind, meaning). "threat" roles trigger intervention; "predator" roles are worth logging;
# "mimic" roles look like bees or wasps and must never trigger anything.
ROLES = {
    "honeybee": ("protected", "Apis: the bees being protected"),
    "bumblebee": ("benign", "Bombus"),
    "other bee": ("benign", "every other bee family"),
    "hornet": ("threat", "Vespa: hunts honeybees at the entrance"),
    "yellowjacket": ("threat", "Vespula and Dolichovespula: raid weak hives"),
    "paper wasp": ("predator", "Polistes, Mischocyttarus, Ropalidia"),
    "other wasp": ("benign", "remaining Hymenoptera: solitary and parasitoid wasps, sawflies, gall wasps"),
    "ant": ("predator", "Formicidae: raid weak hives"),
    "beewolf": ("predator", "Philanthus: hunts honeybees"),
    "robber fly": ("predator", "Asilidae: take bees in flight"),
    "mantis": ("predator", "Mantodea: ambush at entrances"),
    "hoverfly": ("mimic", "Syrphidae: bee and wasp mimics"),
    "other insect": ("benign", "anything outside the set"),
}


def role(genus, family, order):
    """Role for one species, from its genus, family and order."""
    if genus == "Apis":
        return "honeybee"
    if genus == "Bombus":
        return "bumblebee"
    if family in BEE_FAMILIES:
        return "other bee"
    if genus == "Vespa":
        return "hornet"
    if genus in ("Vespula", "Dolichovespula"):
        return "yellowjacket"
    if genus in ("Polistes", "Mischocyttarus", "Ropalidia"):
        return "paper wasp"
    if genus == "Philanthus":
        return "beewolf"
    if family == "Formicidae":
        return "ant"
    if family == "Asilidae":
        return "robber fly"
    if family == "Syrphidae":
        return "hoverfly"
    if order == "Mantodea":
        return "mantis"
    if order == "Hymenoptera":
        return "other wasp"
    return "other insect"


def record_role(c):
    """Role for a classes.json / iNat2021 category record."""
    return role(c["name"].split()[0], c.get("family"), c.get("order"))


def roles_json(classes):
    """The roles.json shipped beside labels.txt: roles, their kind, and the class indices in each."""
    members = {r: [] for r in ROLES}
    for i, c in enumerate(classes):
        members[record_role(c)].append(i)
    return {"rule": "role probability = sum of its species' probabilities; act on roles, smoothed over frames",
            "roles": {r: {"kind": k, "meaning": m, "classes": members[r]} for r, (k, m) in ROLES.items() if members[r]}}


if __name__ == "__main__":
    import sys
    classes = json.load(open(sys.argv[1], encoding="utf-8"))
    print(json.dumps(roles_json(classes), indent=1))
