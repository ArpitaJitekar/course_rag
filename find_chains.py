import sys, inspect, pkgutil

def find_target(pkg_name, target):
    import importlib
    try:
        pkg = importlib.import_module(pkg_name)
    except:
        return
    for importer, modname, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):
        try:
            m = importlib.import_module(modname)
            if hasattr(m, target):
                print(f"Found {target} in {modname}")
        except Exception:
            pass

for p in ['langchain', 'langchain_core', 'langchain_community', 'langchain_classic']:
    print(f"Searching {p} for create_retrieval_chain...")
    find_target(p, 'create_retrieval_chain')
    print(f"Searching {p} for create_stuff_documents_chain...")
    find_target(p, 'create_stuff_documents_chain')
