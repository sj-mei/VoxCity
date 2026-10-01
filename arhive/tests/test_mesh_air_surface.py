"""Air-adjacent and model-top surfaces, independent of legacy mesh modes."""
import numpy as np
from voxcity.geoprocessor.mesh import create_air_surface_mesh, create_voxel_mesh


def test_ground_only_exposes_top_and_preserves_materials():
    grid = np.zeros((2, 3, 3), dtype=int)
    grid[:, :, 0] = -1
    grid[:, :, 1] = 12
    mesh = create_air_surface_mesh(grid, meshsize=2.0)
    assert len(mesh.faces) == 12
    assert np.allclose(mesh.face_normals, [0, 0, 1])
    assert np.allclose(mesh.triangles_center[:, 2], 4.0)
    assert np.all(mesh.metadata['face_voxel_class'] == 12)


def test_selected_material_does_not_expose_contacts_with_other_solids():
    grid = np.full((3, 3, 3), -1)
    grid[1, 1, 1] = -3
    grid[1, 2, 1] = 0
    grid[2, 1, 1] = -2
    mesh = create_air_surface_mesh(grid, class_id=-3)
    assert len(mesh.faces) == 2
    assert np.allclose(mesh.face_normals, [1, 0, 0])
    assert np.all(mesh.metadata['face_voxel_class'] == -3)


def test_solid_model_retains_roof_but_excludes_side_and_bottom_cuts():
    grid = np.full((2, 2, 2), -3)
    mesh = create_air_surface_mesh(grid)
    assert len(mesh.faces) == 8
    assert np.allclose(mesh.face_normals, [0, 0, 1])
    assert np.allclose(mesh.triangles_center[:, 2], 2.0)
    assert np.all(mesh.metadata["face_voxel_class"] == -3)
    assert create_voxel_mesh(grid, -3, mesh_type='open_air') is not None
    assert create_air_surface_mesh(np.zeros((2, 2, 2))) is None
    assert create_air_surface_mesh(grid, class_id=0) is None


def test_air_cavity_is_retained():
    grid = np.full((3, 3, 3), -1)
    grid[1, 1, 1] = 0
    mesh = create_air_surface_mesh(grid)
    assert len(mesh.faces) == 30  # 12 cavity triangles plus 18 top triangles
    assert np.all(mesh.metadata['face_voxel_class'] == -1)


def test_buried_building_has_no_exposed_surface():
    grid = np.full((3, 3, 3), -1)
    grid[1, 1, 1] = -3
    assert create_air_surface_mesh(grid, class_id=-3) is None


def test_top_and_lower_roofs_are_both_retained():
    grid = np.zeros((3, 4, 3), dtype=int)
    grid[:, :, 0] = -1
    grid[1, 1, 1:] = -3
    grid[1, 2, 1] = -3
    mesh = create_air_surface_mesh(grid, class_id=-3)
    roof = np.all(np.isclose(mesh.face_normals, [0, 0, 1]), axis=1)
    assert roof.sum() == 4
    assert set(mesh.triangles_center[roof, 2]) == {2.0, 3.0}
