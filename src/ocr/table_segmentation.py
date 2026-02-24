import numpy as np

def get_box_center_x(bbox):
    xs = [p[0] for p in bbox]
    return sum(xs) / len(xs)

def detect_columns(lines, x_threshold=40):
    """
    Détecte les colonnes du tableau à partir des bounding boxes
    """
    x_centers = []

    for line in lines:
        for bbox, text, conf in line["items"]:
            x_centers.append(get_box_center_x(bbox))

    x_centers = sorted(x_centers)

    columns = []
    for x in x_centers:
        placed = False
        for col in columns:
            if abs(col["x"] - x) < x_threshold:
                col["values"].append(x)
                col["x"] = np.mean(col["values"])
                placed = True
                break
        if not placed:
            columns.append({"x": x, "values": [x]})

        # Trier colonnes de gauche à droite
        columns.sort(key=lambda c: c["x"])

        return columns
    

def assign_boxes_to_columns(lines, columns):
    """
    Assigne chaque bounding box à une colonne détectée
    """
    table = []

    for line in lines:
        row = [""] * len(columns)

        for bbox, text, conf in line["items"]:
            x = get_box_center_x(bbox)

            # Trouver la colonne la plus proche
            col_idx = min(
                range(len(columns)),
                key=lambda i: abs(columns[i]["x"] - x)
            )

            row[col_idx] = text
        
        table.append(row)

    return table