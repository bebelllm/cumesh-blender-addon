# CuMesh Remesh — Blender Addon

Addon Blender qui appelle [CuMesh](https://github.com/JeffreyXiang/CuMesh)
(`remesh_narrow_band_dc`, GPU, Dual Contouring bande étroite) via un
sous-process Python externe.

## Architecture

- `cumesh_blender_addon/__init__.py` — addon Blender (panneau +
  opérateur, export/import PLY)
- `cumesh_blender_addon/cumesh_worker.py` — script autonome exécuté
  dans un venv externe qui a seulement `torch` + `cumesh` (PLY I/O fait
  en pur `numpy`, aucune autre dépendance requise)
- Échange Blender ↔ worker via fichiers PLY temporaires (binaires
  little-endian)

Le mesh actif est remaillé dans un **nouvel objet** (non-destructif) ;
l'original est masqué par défaut.

## Installation

### 1. Créer un venv externe avec CuMesh

Prérequis :
- Python ≥ 3.8
- CUDA Toolkit ≥ 12.4 (13.0 fonctionne aussi)
- GPU NVIDIA
- **MSVC (Visual Studio Build Tools)** sur Windows — `cl.exe` doit être
  dans le PATH. Lancez les commandes depuis une **"x64 Native Tools
  Command Prompt for VS 2022"** (ou l'équivalent PowerShell) pour que ça
  soit le cas automatiquement.

```powershell
python -m venv C:\envs\cumesh
C:\envs\cumesh\Scripts\activate

# torch CUDA (adapter l'index-url à votre version de CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# cumesh (compilation depuis les sources — sous-modules obligatoires)
cd C:\envs\cumesh
git clone --recursive https://github.com/JeffreyXiang/CuMesh.git
cd CuMesh
pip install . --no-build-isolation
```

⚠️ **Important — architecture GPU**. Sur les versions récentes de PyTorch,
la liste d'archs CUDA par défaut pour les extensions tierces démarre à
`sm_80`, ce qui **exclut les cartes Turing (sm_75, ex. RTX 2080 Ti)** et
certaines archs plus anciennes. Si votre GPU n'est pas couvert, le build
réussit mais cumesh crashera au runtime avec
`CUDA error: no kernel image is available for execution on the device`.
Dans ce cas, forcez la liste d'archs avant de (re)compiler cumesh, par
exemple pour une 2080 Ti :

```powershell
$env:TORCH_CUDA_ARCH_LIST = "7.5"
pip install . --no-build-isolation --no-cache-dir
```

Valeurs usuelles : `6.1` (Pascal), `7.0` (V100), `7.5` (Turing — 20xx,
T4), `8.0` (A100), `8.6` (Ampere — 30xx), `8.9` (Ada — 40xx),
`9.0` (Hopper — H100), `12.0` (Blackwell — 50xx). Multi-arch :
`"7.5;8.6;8.9"`. Ajouter `+PTX` (ex. `"7.5+PTX"`) pour embarquer du PTX
et permettre la JIT sur des GPU plus récents que ceux listés.

Vérification rapide :

```powershell
C:\envs\cumesh\Scripts\python.exe -c "import torch, cumesh; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Doit afficher `True <nom de votre GPU>`.

### 2. Installer l'addon dans Blender

Testé sur Blender 5.0 (fonctionne également sur 4.x).

1. Zipper le dossier `cumesh_blender_addon/` (le zip doit contenir un
   dossier du même nom avec `__init__.py` et `cumesh_worker.py` à
   l'intérieur).
2. Blender → `Edit` → `Preferences` → `Add-ons` → `Install from Disk` →
   choisir le zip.
3. Activer **CuMesh Remesh**.
4. Déplier l'addon et renseigner **External Python** avec le chemin
   complet du `python.exe` du venv :
   `C:\envs\cumesh\Scripts\python.exe`
5. Si "Auto-Save Preferences" est désactivé, cliquez sur le menu
   hamburger en bas à gauche de la fenêtre des préférences →
   **Save Preferences**.

### 3. Utilisation

1. Sélectionner un mesh.
2. `N` → onglet **CuMesh**.
3. Régler `Resolution`, `Band`, `Project Back`, `Scale Padding`.
4. Cliquer **CuMesh Remesh**.

Un nouvel objet `<nom>_cumesh` apparaît, l'original est masqué.

## Paramètres

| Paramètre       | Description                                                             | Défaut |
|-----------------|-------------------------------------------------------------------------|--------|
| Resolution      | Résolution du grid DC sur le plus grand axe de la bbox                 | 128    |
| Band            | Largeur de la bande étroite (en voxels)                                 | 1      |
| Project Back    | Force de projection des sommets DC sur la surface d'origine (0..1)      | 0.9    |
| Scale Padding   | Multiplicateur de la bbox pour éviter de couper la surface              | 1.05   |
| Hide Source     | Masquer l'objet original après remesh                                    | ✓      |
| Verbose         | Logs détaillés dans la console système de Blender                       | ✗      |

## Limitations / Notes

- **Pas un vrai "modifier"** : Blender n'autorise pas les addons Python à
  ajouter des modifiers dans la pile. C'est un opérateur, pas une entrée
  dans la stack des modifiers.
- Si vous changez les paramètres, il faut : réafficher l'original, le
  re-sélectionner, supprimer l'objet `_cumesh` précédent, relancer.
- CuMesh nécessite un GPU NVIDIA compatible CUDA 12.4+.
- L'échange via PLY ne transporte que positions + topologie. UVs,
  normales custom et vertex colors de l'original ne sont pas propagés au
  mesh remaillé (ce qui est cohérent avec un remesh basé sur un champ de
  distance, qui recrée une topologie entièrement nouvelle).
- Transforms : le mesh est exporté dans son repère local ; la matrice
  world de l'original est ré-appliquée au nouvel objet.
- Pour lire les logs détaillés côté Blender :
  `Window` → `Toggle System Console` (Windows).

## Dépannage

| Message | Cause probable | Correctif |
|---|---|---|
| `CuMesh: 'External Python' is empty` | Le champ n'est pas rempli en préférences | Éditer les prefs de l'addon + Save Preferences |
| `CuMesh: External Python not found at: ...` | Le chemin existe mais le fichier n'est pas là | Vérifier le chemin du `python.exe` du venv |
| `ERROR: cumesh not importable` | Le venv n'a pas `cumesh` | Installer cumesh dans **ce** venv (cf. §1) |
| `ERROR: CUDA is not available in this torch build` | Torch CPU-only dans le venv | Réinstaller torch avec la wheel CUDA |
| `CUDA error: no kernel image is available for execution on the device` | cumesh compilé sans l'arch du GPU | Rebuild cumesh avec `TORCH_CUDA_ARCH_LIST` (cf. §1) |
| `CuMesh worker failed (code N)` + pas de détails | Voir la console système de Blender pour stdout/stderr du sous-process | `Window` → `Toggle System Console` |

## Licence

L'addon lui-même est publié sous licence MIT. CuMesh (la dépendance
externe) est publié par ses auteurs sous MIT également.
