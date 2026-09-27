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


# ── two ways to find a parent ───────────────────────────────────

def get_spatial_parent(obj):
    """Parent via IfcRelAggregates (spatial hierarchy: Space→Storey→Building…)."""
    try:
        for rel in getattr(obj, "Decomposes", ()):
            if rel.is_a("IfcRelAggregates"):
                return rel.RelatingObject
    except Exception:
        pass
    return None


def get_container(elem):
    """Parent via IfcRelContainedInSpatialStructure (element→spatial)."""
    try:
        return ifcopenshell.util.element.get_container(elem)
    except Exception:
        return None


# ── walk up both chains to find the storey ──────────────────────

def find_storey(obj):
    """Walk up containment + aggregation until we hit an IfcBuildingStorey."""
    cur = obj
    visited = set()
    while cur:
        if cur.is_a("IfcBuildingStorey"):
            return cur
        gid = attr(cur, "GlobalId")
        if not gid or gid in visited:
            return None
        visited.add(gid)
        cur = get_container(cur) or get_spatial_parent(cur)
    return None


def get_storey_name(obj):
    s = find_storey(obj)
    return attr(s, "Name") if s else None


def get_storey_guid(obj):
    s = find_storey(obj)
    return attr(s, "GlobalId") if s else None


def get_direct_container_guid(elem):
    c = get_container(elem)
    return attr(c, "GlobalId") if c else None


# ── spatial tree ────────────────────────────────────────────────

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

            parent = get_spatial_parent(obj)
            pgid = attr(parent, "GlobalId") if parent else None

            node = {
                "guid": gid,
                "type": obj.is_a(),
                "name": attr(obj, "Name"),
                "longName": attr(obj, "LongName"), 
                "parentGuid": pgid,
                "children": [],
                "elements": [],
                "data": get_psets_flat(obj),
            }

            # ── add storey info to spaces (and anything below a storey) ──
            if obj.is_a("IfcBuildingStorey"):
                node["storey"] = attr(obj, "Name")
                node["storeyGuid"] = gid
            else:
                node["storey"] = get_storey_name(obj)
                node["storeyGuid"] = get_storey_guid(obj)

            spatial[gid] = node

    for gid, node in spatial.items():
        pg = node["parentGuid"]
        if pg and pg in spatial:
            spatial[pg]["children"].append(gid)
        else:
            roots.append(gid)

    return spatial, roots


# ── elements ────────────────────────────────────────────────────

def collect_elements(model, spatial):
    elements = {}
    seen = set()

    for elem in safe_by_type(model, "IfcElement"):
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
            "storeyGuid": get_storey_guid(elem),
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