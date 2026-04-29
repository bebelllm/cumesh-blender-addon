# CuMesh Remesh — Blender Addon

Addon Blender multi-backend pour le **remaillage** et la **réparation**
de meshes. Chaque backend tourne dans un sous-process Python externe ;
les opérations sont **non-destructives** (un nouvel objet est créé,
l'original est conservé) et **chaînables** (on peut empiler plusieurs
backends).

## Backends inclus

| Backend         | Bibliothèque                                                                              | GPU/CPU  | Pour quoi faire                                                  |
|-----------------|-------------------------------------------------------------------------------------------|----------|------------------------------------------------------------------|
| **CuMesh**      | [JeffreyXiang/CuMesh](https://github.com/JeffreyXiang/CuMesh) — `remesh_narrow_band_dc`   | **GPU**  | Remesh voxel narrow-band DC, rapide, décimation depuis dense     |
| **Manifold**    | [elalish/manifold](https://github.com/elalish/manifold) (`manifold3d`)                    | CPU      | CSG / refine / smooth, **exige un input watertight**             |
| **PyMeshFix**   | [pyvista/pymeshfix](https://github.com/pyvista/pymeshfix) (algo Marco Attene)             | CPU      | **Réparation** : bouche les trous, élimine non-manifold/intersections |
| **PyMeshLab**   | [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) — `meshing_isotropic_explicit_remeshing` (GPL) | CPU      | Remesh isotrope (triangles équilatéraux), zéro artefact          |
| **Instant Meshes** | [pynanoinstantmeshes](https://pypi.org/project/pynanoinstantmeshes/) (Wenzel Jakob)    | CPU      | Remesh field-aligned (quads ou tris), idéal sur petites entrées  |

Tous les backends partagent le même interpréteur Python (configuré dans
les préférences de l'addon) et chaque worker vérifie ses propres
dépendances avec un message d'install clair s'il en manque une.

## Architecture

```
Blender (addon)
   │
   │   bpy.ops.cumesh.<backend>_remesh
   ↓
exporte la mesh active en PLY temporaire
   │
   ↓
spawn  python.exe <backend>_worker.py --input ... --output ...
                  │
                  │   imports lazy : torch+cumesh / manifold3d /
                  │   pymeshfix / pymeshlab / pynanoinstantmeshes
                  ↓
        traitement, écrit le PLY de sortie
   │
   ↓
ré-importe en remplaçant la mesh d'un nouvel objet, lié à la source
via la PointerProperty `cumesh_source` + `cumesh_backend` string
```

Fichiers :
- `cumesh_blender_addon/__init__.py` — addon Blender (panneau, opérateurs, helpers)
- `cumesh_blender_addon/cumesh_worker.py` — worker CuMesh
- `cumesh_blender_addon/manifold_worker.py` — worker Manifold
- `cumesh_blender_addon/meshfix_worker.py` — worker PyMeshFix
- `cumesh_blender_addon/pymeshlab_worker.py` — worker PyMeshLab
- `cumesh_blender_addon/instant_meshes_worker.py` — worker Instant Meshes

## Installation

### 1. Créer un venv externe

Prérequis :
- Python ≥ 3.10
- **GPU NVIDIA + CUDA Toolkit** (12.4 ou supérieur) — uniquement pour CuMesh
- **MSVC Build Tools** sur Windows — uniquement pour compiler CuMesh depuis les sources

```powershell
python -m venv C:\envs\cumesh
C:\envs\cumesh\Scripts\activate

# torch CUDA (adapter l'index-url à votre version de CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Backends optionnels (installer ceux dont vous avez besoin)
pip install manifold3d            # Manifold backend
pip install pymeshfix             # PyMeshFix backend
pip install pymeshlab             # PyMeshLab backend (GPL)
pip install pynanoinstantmeshes   # Instant Meshes backend
```

### 2. Installer CuMesh (optionnel mais souvent utile)

CuMesh doit être compilé depuis les sources avec MSVC. Lancez les
commandes depuis une **"x64 Native Tools Command Prompt for VS 2022"**.

```powershell
cd C:\envs\cumesh
git clone --recursive https://github.com/JeffreyXiang/CuMesh.git
cd CuMesh
pip install . --no-build-isolation
```

⚠️ **Architecture GPU**. Les versions récentes de PyTorch démarrent la
liste d'archs CUDA à `sm_80`, ce qui **exclut Turing (sm_75, RTX 2080
Ti)** et plus ancien. Si votre GPU n'est pas couvert, le build réussit
mais cumesh crashera au runtime avec
`CUDA error: no kernel image is available for execution on the device`.
Forcez la liste d'archs avant de (re)compiler :

```powershell
$env:TORCH_CUDA_ARCH_LIST = "7.5"   # Turing
# 8.6 = Ampere (30xx), 8.9 = Ada (40xx), 9.0 = Hopper, 12.0 = Blackwell (50xx)
pip install . --no-build-isolation --no-cache-dir
```

Vérification :
```powershell
C:\envs\cumesh\Scripts\python.exe -c "import torch, cumesh; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Installer l'addon dans Blender

Testé sur Blender 4.x et 5.0.

1. Zipper le dossier `cumesh_blender_addon/` (le zip doit contenir un
   dossier du même nom avec les `*.py` à l'intérieur).
2. Blender → `Edit` → `Preferences` → `Add-ons` → `Install from Disk` →
   choisir le zip.
3. Activer **CuMesh Remesh**.
4. Déplier l'addon et renseigner **External Python** avec le chemin
   complet du `python.exe` du venv : `C:\envs\cumesh\Scripts\python.exe`
5. Save Preferences.

## Utilisation

`N` (sidebar) → onglet **CuMesh**. Le panneau contient une section par
backend, chacune avec ses propres paramètres et son bouton.

### Workflow type sur un scan / mesh sale

1. Sélectionner la source.
2. **PyMeshFix Repair** (`Remove Smallest = ON`) → crée
   `<nom>_meshfix`, watertight, 1 composant.
3. Sur le résultat meshfix, choisir un remailleur :
   - **PyMeshLab Isotropic** (recommandé pour qualité topologique : 1
     composant, 0 non-manifold, triangles équilatéraux)
   - **CuMesh** (`band=3`, `project_back=0.9`) pour une décimation GPU
     rapide
   - **Instant Meshes** pour une topologie field-aligned (quads ou tris)
4. **Manifold Remesh** optionnel pour valider l'étanchéité ou subdiviser
   (`Refine`).

### Section Linked Result

Une fois sur un objet généré par n'importe quel backend, le panneau
affiche un encart **Linked Result** avec :

- **Refresh CuMesh / Refresh \<backend\>** → re-run sur la source liée,
  remplace la mesh in-place (préserve l'identité de l'objet, ses
  modifiers, parents, materials)
- **Edit Source** → réaffiche et sélectionne la source
- **Toggle Source / Remesh** → bascule visibilité source ↔ résultat
- Les paramètres utilisés au dernier bake sont affichés en bas

### Section Topology Stats

Bouton **Analyse Topology** : compte composants connexes, arêtes
non-manifold, arêtes de bord, faces d'aire nulle. Stocké en custom
properties sur l'objet et persistant au `.blend`. Auto-appelé après
Remesh/Refresh tant que la mesh ≤ 500K verts (au-delà c'est manuel pour
ne pas freezer l'UI).

## Paramètres par backend

### CuMesh
| Paramètre       | Description                                                            | Défaut |
|-----------------|------------------------------------------------------------------------|--------|
| Resolution      | Résolution du grid DC sur le plus grand axe de la bbox                | 128    |
| Band            | Largeur de la bande étroite (en voxels). **3 = bien plus propre** que 1 | 1      |
| Project Back    | Force de projection des sommets DC sur la surface d'origine (0..1)     | 0.9    |
| Scale Padding   | Multiplicateur de la bbox                                              | 1.05   |

### Manifold
| Paramètre        | Description                                          | Défaut |
|------------------|------------------------------------------------------|--------|
| Refine           | Subdivisions d'arêtes (0 = pas de raffinement)       | 0      |
| Smooth by Normals| Lissage normales-aware après construction (peut crash sur certaines géo) | OFF    |

### PyMeshFix
| Paramètre         | Description                                              | Défaut |
|-------------------|----------------------------------------------------------|--------|
| Join Components   | Fusionner tous les composants en un seul mesh            | OFF    |
| Remove Smallest   | Garder uniquement le plus gros composant                 | ON     |

### PyMeshLab
| Paramètre        | Description                                                  | Défaut |
|------------------|--------------------------------------------------------------|--------|
| Target Edge Length | Longueur d'arête cible (0 = auto = 1% diagonale bbox)      | 0      |
| Iterations       | Passes de remaillage                                         | 3      |
| Smooth Pass      | Lissage par itération                                        | ON     |
| Reproject        | Reprojection sur la surface originale après chaque passe     | ON     |

### Instant Meshes
| Paramètre         | Description                                              | Défaut |
|-------------------|----------------------------------------------------------|--------|
| Target Vertices   | Nombre approximatif de sommets en sortie                 | 10000  |
| Quads             | Sortie en quads (sinon triangles)                        | OFF    |
| Smooth Iterations | Lissage                                                  | 2      |
| Crease Angle      | Seuil dièdre pour les arêtes vives (0 = désactivé)       | 0.0    |
| Align Boundaries  | Aligner sur les bords du mesh                            | OFF    |

## Limitations / Notes

- **Ce n'est pas un Modifier** : Blender n'autorise pas les addons
  Python à ajouter des modifiers dans la stack. C'est un opérateur, pas
  une entrée dans la stack des modifiers. Le **Refresh** est notre
  équivalent "live" — re-run en in-place sur la source liée.
- **L'échange via PLY** ne transporte que positions + topologie. UVs,
  normales custom et vertex colors de l'original ne sont pas propagés.
- **Transforms** : la mesh est exportée en local space ; la matrix
  world de l'original est ré-appliquée au nouvel objet. Pour des
  sources à scale très non-uniforme (ex. `Scale Z = 0.05`),
  pré-appliquer le scale (`Ctrl+A → Scale`) donne souvent de meilleurs
  résultats.
- **Manifold** rejette les inputs non watertight — l'addon fait un
  pre-check et renvoie un message clair pour rediriger vers PyMeshFix.
- **Instant Meshes** peut crasher (STATUS_STACK_BUFFER_OVERRUN) sur des
  inputs très denses. Si ça arrive : baisser `Target Vertices`,
  pré-cleaner avec PyMeshFix, ou utiliser PyMeshLab.

## Dépannage

| Message | Cause probable | Correctif |
|---|---|---|
| `'External Python' is empty` | Le champ n'est pas rempli en préférences | Éditer les prefs de l'addon + Save Preferences |
| `External Python not found at: ...` | Le chemin existe mais le fichier n'est pas là | Vérifier le chemin du `python.exe` du venv |
| `<lib> not importable in this Python interpreter` | Le venv n'a pas la lib correspondante | Installer la lib dans **ce** venv (cf. §1) |
| `CUDA is not available in this torch build` | Torch CPU-only dans le venv | Réinstaller torch avec la wheel CUDA |
| `CUDA error: no kernel image is available...` | cumesh compilé sans l'arch du GPU | Rebuild cumesh avec `TORCH_CUDA_ARCH_LIST` (cf. §2) |
| `Manifold requires watertight input` | La source a des trous / non-manifold | Run **PyMeshFix Repair** d'abord, puis **Manifold** sur le résultat |
| `pynanoinstantmeshes C++ crashed during position-field optimization` | Input trop dense ou bruité | Baisser `Target Vertices`, ou utiliser **PyMeshLab Isotropic** |
| `worker timed out (30 min)` | L'algo n'a pas convergé | Pre-décimer la source, baisser les paramètres |

Pour les détails côté Blender :
`Window` → `Toggle System Console` (Windows).

## Licence

Addon publié sous **MIT**.

Les bibliothèques externes ont leurs propres licences :
- CuMesh : MIT
- Manifold (manifold3d) : Apache 2.0
- PyMeshFix : MIT
- PyMeshLab : **GPL** (hérité de MeshLab) — usage commercial à vérifier
- Instant Meshes (pynanoinstantmeshes) : **GPL** (algo Wenzel Jakob)
