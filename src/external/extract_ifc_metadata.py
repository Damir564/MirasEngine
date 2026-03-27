import sys
import json
import ifcopenshell
import ifcopenshell.util.element


def attr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def safe_by_type(model, t):
    try:
        return model.by_type(t)
    except RuntimeError:
        return []


def safe_value(v):
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): safe_value(val) for k, val in v.items()}
    return str(v)


def get_psets_flat(elem):
    result = {}
    try:
        psets = ifcopenshell.util.element.get_psets(elem)
        for pset_name, props in psets.items():
            if isinstance(props, dict):
                for k, v in props.items():
                    if k == "id":
                        continue
                    result[f"{pset_name}.{k}"] = safe_value(v)
    except Exception:
        pass
    return result


def get_type_info(elem):
    try:
        t = ifcopenshell.util.element.get_type(elem)
        if not t:
            return None
        return {
            "guid": attr(t, "GlobalId"),
            "type": t.is_a(),
            "name": attr(t, "Name"),
            "predefinedType": attr(t, "PredefinedType"),
        }
    except Exception:
        return None


def get_container(elem):
    try:
        return ifcopenshell.util.element.get_container(elem)
    except Exception:
        return None


def get_storey_name(elem):
    cur = elem
    visited = set()
    while True:
        c = get_container(cur)
        if not c:
            return None
        gid = attr(c, "GlobalId")
        if gid in visited:
            return None
        visited.add(gid)
        if c.is_a() == "IfcBuildingStorey":
            return attr(c, "Name")
        cur = c


def get_direct_container_guid(elem):
    c = get_container(elem)
    if c:
        return attr(c, "GlobalId")
    return None


def collect_spatial(model):
    spatial = {}
    roots = []
    seen = set()

    types = [
        "IfcProject", "IfcSite", "IfcBuilding",
        "IfcBuildingStorey", "IfcSpace"
    ]

    for t in types:
        for obj in safe_by_type(model, t):
            gid = attr(obj, "GlobalId")
            if not gid or gid in seen:
                continue
            seen.add(gid)

            parent = get_container(obj)
            pgid = attr(parent, "GlobalId") if parent else None

            spatial[gid] = {
                "guid": gid,
                "type": obj.is_a(),
                "name": attr(obj, "Name"),
                "parentGuid": pgid,
                "children": [],
                "elements": []
            }

    for gid, node in spatial.items():
        pg = node["parentGuid"]
        if pg and pg in spatial:
            spatial[pg]["children"].append(gid)
        else:
            roots.append(gid)

    return spatial, roots


def collect_elements(model, spatial):
    elements = {}

    candidates = []
    candidates.extend(safe_by_type(model, "IfcElement"))
    # candidates.extend(safe_by_type(model, "IfcSpace"))  # ADD THIS

    seen = set()

    for elem in candidates:
        guid = attr(elem, "GlobalId", None)
        if not guid or guid in seen:
            continue
        seen.add(guid)

        container_guid = get_direct_container_guid(elem)

        elements[guid] = {
            "guid": guid,
            "type": elem.is_a(),
            "name": attr(elem, "Name", None),
            "objectType": attr(elem, "ObjectType", None),
            "tag": attr(elem, "Tag", None),
            "storey": get_storey_name(elem),
            "parentSpatialGuid": container_guid,
            "typeInfo": get_type_info(elem),
            "data": get_psets_flat(elem)
        }

        if container_guid and container_guid in spatial:
            spatial[container_guid]["elements"].append(guid)

    return elements


def main():
    if len(sys.argv) < 3:
        print("usage: extract_ifc_hierarchy.py input.ifc output.json")
        sys.exit(1)

    ifc_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"[IFC] opening {ifc_path}")
    model = ifcopenshell.open(ifc_path)
    print(f"[IFC] schema = {model.schema}")

    spatial, roots = collect_spatial(model)
    elements = collect_elements(model, spatial)

    result = {
        "schema": model.schema,
        "roots": roots,
        "spatial": spatial,
        "elements": elements
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[OK] {len(elements)} elements, {len(spatial)} spatial nodes")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()