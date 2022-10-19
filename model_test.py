import yaml

cfg = {}
with open("config/CrawlerPPO.yaml", "r") as f:
    cfg["model"] = yaml.safe_load(f)

print(cfg["model"])
