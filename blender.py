import bpy
import json
import bmesh
from mathutils import Vector, Quaternion
import math

class BlenderEnvironmentBuilder:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def create_ground_plane(self, thickness=0.0508):
        """
        Builds a wooden ground plane from corners, extruded downward by `thickness` meters.
        """
        corners = self.data["ground_plane"]["corners"]
        print(f"Ground plane corners: {corners}")

        # Convert to Vector3 and drop the z=0
        verts2d = [Vector((c[0], c[1], 0.0)) for c in corners]
        print(f"Vertices: {verts2d}")

        # Create mesh and bmesh
        mesh = bpy.data.meshes.new("GroundPlaneMesh")
        bm   = bmesh.new()

        # Create vertices
        bm_verts = [bm.verts.new(v) for v in verts2d]
        bm.verts.ensure_lookup_table()

        # Create face with correct winding order
        try:
            face = bm.faces.new(bm_verts)
            print("✅ Face created successfully")
        except Exception as e:
            print(f"❌ Face creation failed: {e}")
            # Try with reversed winding
            try:
                face = bm.faces.new(reversed(bm_verts))
                print("✅ Face created with reversed winding")
            except Exception as e2:
                print(f"❌ Face creation failed both ways: {e2}")
                bm.free()
                return None

        # Recalculate normals
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

        # Extrude downward
        if len(bm.faces) > 0:
            ret = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
            verts_extruded = [e for e in ret['geom'] if isinstance(e, bmesh.types.BMVert)]
            bmesh.ops.translate(bm, verts=verts_extruded, vec=Vector((0, 0, -thickness)))

        # Finalize mesh
        bm.to_mesh(mesh)
        bm.free()

        # Create object
        obj = bpy.data.objects.new("GroundPlane", mesh)
        bpy.context.collection.objects.link(obj)

        # Optionally assign a wood material
        mat = bpy.data.materials.get("WoodMaterial") or bpy.data.materials.new("WoodMaterial")
        # simple diffuse brown
        mat.diffuse_color = (0.6, 0.4, 0.2, 1.0)
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

        print(f"✅ Ground plane created with {len(mesh.vertices)} vertices, {len(mesh.polygons)} faces")
        return obj


    def create_sensor_cube(self, size=None):
        """
        Adds a cube of side `size` at the sensor position with the correct orientation.
        If size is None, uses the size of marker ID 100 from the JSON data.
        """
        pos  = self.data["sensor"]["position"]
        quat = self.data["sensor"]["orientation_quat"]  # [w,x,y,z]

        # Get size from marker ID 100 if not specified
        if size is None:
            marker_sizes = self.data.get("marker_sizes", {})
            size = float(marker_sizes.get("100", 0.05))  # Default to 0.05 if not found
            print(f"Using marker 100 size: {size}m")

        # Create cube
        px, py, pz = pos
        bl_x = -py
        bl_y = -px
        bl_z = -pz
        bpy.ops.mesh.primitive_cube_add(size=size, location=(bl_x, bl_y, bl_z))
        cube = bpy.context.active_object
        cube.name = "SensorCube"

        # Apply orientation: preserve only yaw (rotation about Z)
        # Build quaternion and invert
        q = Quaternion(quat)
        q_inv = q.inverted()
        # Extract Euler angles in XYZ order: roll (X), pitch (Y), yaw (Z)
        e = q_inv.to_euler('XYZ')
        yaw = e.z
        # Set cube rotation to zero roll/pitch and keep yaw
        cube.rotation_mode = 'XYZ'
        cube.rotation_euler = (0.0, 0.0, yaw)

        # Assign a simple bright material so it stands out
        mat = bpy.data.materials.get("SensorMat") or bpy.data.materials.new("SensorMat")
        mat.use_nodes = True

        # Clear existing nodes and create new ones
        mat.node_tree.nodes.clear()

        # Create emission shader node
        emission_node = mat.node_tree.nodes.new(type="ShaderNodeEmission")
        emission_node.inputs["Color"].default_value = (0.0, 0.6, 1.0, 1.0)  # Bright cyan
        emission_node.inputs["Strength"].default_value = 2.0

        # Create output node
        output_node = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")

        # Connect emission to output
        mat.node_tree.links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

        # Assign material to cube
        if not cube.data.materials:
            cube.data.materials.append(mat)
        else:
            cube.data.materials[0] = mat

        return cube

    def build(self, save_file=False, quit_blender=False):
        # Clear all existing objects from the scene (including default cube)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        bpy.ops.object.empty_add(type='ARROWS', location=(0,0,0))

        # Alternative method if the above doesn't work:
        # for obj in bpy.data.objects:
        #     bpy.data.objects.remove(obj, do_unlink=True)

        print("✅ Cleared all existing objects from scene")

        # Create ground plane (try simple method first, fallback to complex)
        # Create ground plane & sensor cube
        plane = self.create_ground_plane()
        cube  = self.create_sensor_cube()

        # Parent both under a root empty so we can flip the entire scene
        root = bpy.data.objects.new("SceneRoot", None)
        bpy.context.collection.objects.link(root)
        plane.parent = root
        cube.parent  = root

        # Rotate root: 180° around axis [1, -1, 0] normalized
        axis = (1.0/math.sqrt(2.0), -1.0/math.sqrt(2.0), 0.0)
        root.rotation_mode        = 'AXIS_ANGLE'
        root.rotation_axis_angle  = (math.pi, axis[0], axis[1], axis[2])
        print("✅ Environment built: ground plane + sensor cube.")

        # Save the blend file
        if save_file:
            blend_filename = "aruco_scene.blend"
            bpy.ops.wm.save_as_mainfile(filepath=blend_filename)
            print(f"✅ Saved scene as: {blend_filename}")

        # Optionally quit Blender
        if quit_blender:
            print("🔄 Quitting Blender...")
            bpy.ops.wm.quit_blender()
        else:
            print("🔄 Blender remains open. Close manually when done.")


# Usage:
# 1. Save this script in Blender’s Text Editor.
# 2. Adjust the path to your JSON file below.
# 3. Run the script (Alt+P).

if __name__ == "__main__":
    builder = BlenderEnvironmentBuilder("test_scene.json")

    # Option 1: Build scene, save file, keep Blender open (default)
    builder.build(save_file=False, quit_blender=False)

    # Option 2: Build scene, save file, and quit Blender automatically
    # builder.build(use_simple_plane=True, save_file=True, quit_blender=True)

    # Option 3: Build scene but don't save file (for testing)
    # builder.build(use_simple_plane=True, save_file=False, quit_blender=False)
