import yaml

with open('data/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
    print(f"Number of names in yaml: {len(data['names'])}")
    print(f"Number of classes in yaml: {data['nc']}")
