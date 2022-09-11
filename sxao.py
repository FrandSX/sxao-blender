bl_info = {
    'name': 'SX Ambient Occlusion',
    'author': 'Jani Kahrama / Secret Exit Ltd.',
    'version': (1, 0, 1),
    'blender': (3, 2, 0),
    'location': 'View3D',
    'description': 'Vertex Ambient Occlusion Tool',
    'doc_url': 'https://www.notion.so/SX-Tools-for-Blender-Documentation-9ad98e239f224624bf98246822a671a6',
    'tracker_url': 'https://github.com/FrandSX/sxtools-blender/issues',
    'category': 'Development',
}


import bpy
import random
import math
import bmesh
import statistics
from mathutils import Vector


# ------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------
class SXAO_sxglobals(object):
    def __init__(self):
        self.refreshInProgress = False
        self.modalStatus = False
        self.listItems = []
        self.listIndices = {}
        self.prevMode = 'OBJECT'
        self.mode = None
        self.modeID = None

        self.prevSelection = []
        self.prevComponentSelection = []

        self.randomseed = 42


    def __del__(self):
        print('SX Tools: Exiting sxglobals')


# ------------------------------------------------------------------------
#    Useful Miscellaneous Functions
# ------------------------------------------------------------------------
class SXAO_utils(object):
    def __init__(self):
        return None


    def find_root_pivot(self, objs):
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_object_bounding_box(objs)
        pivot = ((xmax + xmin)*0.5, (ymax + ymin)*0.5, zmin)

        return pivot


    def get_object_bounding_box(self, objs, local=False):
        bbx_x = []
        bbx_y = []
        bbx_z = []
        for obj in objs:
            if local:
                corners = [Vector(corner) for corner in obj.bound_box]
            else:
                corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

            for corner in corners:
                bbx_x.append(corner[0])
                bbx_y.append(corner[1])
                bbx_z.append(corner[2])
        xmin, xmax = min(bbx_x), max(bbx_x)
        ymin, ymax = min(bbx_y), max(bbx_y)
        zmin, zmax = min(bbx_z), max(bbx_z)

        return xmin, xmax, ymin, ymax, zmin, zmax


    def get_selection_bounding_box(self, objs):
        vert_pos_list = []
        for obj in objs:
            mesh = obj.data
            mat = obj.matrix_world
            for vert in mesh.vertices:
                if vert.select:
                    vert_pos_list.append(mat @ vert.co)

        bbx = [[None, None], [None, None], [None, None]]
        for i, fvPos in enumerate(vert_pos_list):
            # first vert
            if i == 0:
                bbx[0][0] = bbx[0][1] = fvPos[0]
                bbx[1][0] = bbx[1][1] = fvPos[1]
                bbx[2][0] = bbx[2][1] = fvPos[2]
            else:
                for j in range(3):
                    if fvPos[j] < bbx[j][0]:
                        bbx[j][0] = fvPos[j]
                    elif fvPos[j] > bbx[j][1]:
                        bbx[j][1] = fvPos[j]

        return bbx[0][0], bbx[0][1], bbx[1][0], bbx[1][1], bbx[2][0], bbx[2][1]


    def find_safe_mesh_offset(self, obj):
        bias = 0.0001

        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        max_distances = []

        for vert in bm.verts:
            vert_loc = vert.co
            inv_normal = -1.0 * vert.normal.normalized()
            bias_vec = inv_normal * bias
            ray_origin = (vert_loc[0] + bias_vec[0], vert_loc[1] + bias_vec[1], vert_loc[2] + bias_vec[2])

            hit, loc, normal, index = obj.ray_cast(ray_origin, inv_normal)

            if hit:
                dist = Vector((loc[0] - vert_loc[0], loc[1] - vert_loc[1], loc[2] - vert_loc[2])).length
                if dist > 0.0:
                    max_distances.append(dist)

        bm.free
        return min(max_distances)


    def __del__(self):
        print('SX Tools: Exiting utils')


# ------------------------------------------------------------------------
#    Value Generators and Utils
#    NOTE: Switching between EDIT and OBJECT modes is slow.
#          Make sure OBJECT mode is enabled before calling
#          any functions in this class!
# ------------------------------------------------------------------------
class SXAO_generate(object):
    def __init__(self):
        return None


    def ray_randomizer(self, count):
        hemiSphere = [None] * count
        random.seed(sxglobals.randomseed)

        for i in range(count):
            u1 = random.random()
            u2 = random.random()
            r = math.sqrt(u1)
            theta = 2*math.pi*u2

            x = r * math.cos(theta)
            y = r * math.sin(theta)

            hemiSphere[i] = (x, y, math.sqrt(max(0, 1 - u1)))

        return hemiSphere


    def ground_plane(self, size, pos):
        vertArray = []
        faceArray = []
        size *= 0.5

        vert = [(pos[0]-size, pos[1]-size, pos[2])]
        vertArray.extend(vert)
        vert = [(pos[0]+size, pos[1]-size, pos[2])]
        vertArray.extend(vert)
        vert = [(pos[0]-size, pos[1]+size, pos[2])]
        vertArray.extend(vert)
        vert = [(pos[0]+size, pos[1]+size, pos[2])]
        vertArray.extend(vert)

        face = [(0, 1, 3, 2)]
        faceArray.extend(face)

        mesh = bpy.data.meshes.new('groundPlane_mesh')
        groundPlane = bpy.data.objects.new('groundPlane', mesh)
        bpy.context.scene.collection.objects.link(groundPlane)

        mesh.from_pydata(vertArray, [], faceArray)
        mesh.update(calc_edges=True)

        # groundPlane.location = pos
        return groundPlane, mesh


    def thickness_list(self, obj, raycount):

        def dist_hit(vert_id, loc, vertPos, dist_list):
            distanceVec = Vector((loc[0] - vertPos[0], loc[1] - vertPos[1], loc[2] - vertPos[2]))
            dist_list.append(distanceVec.length)

        def thick_hit(vert_id, loc, vertPos, dist_list):
            vert_occ_dict[vert_id] += contribution

        def ray_caster(obj, raycount, vert_dict, hitfunction, raydistance=1.70141e+38):
            hemiSphere = self.ray_randomizer(raycount)

            for vert_id in vert_dict:
                vert_occ_dict[vert_id] = 0.0
                vertLoc = Vector(vert_dict[vert_id][0])
                vertNormal = Vector(vert_dict[vert_id][1])
                bias = 0.001

                # Invert normal to cast inside object
                invNormal = tuple([-1*x for x in vertNormal])

                # Raycast for bias
                hit, loc, normal, index = obj.ray_cast(vertLoc, invNormal, distance=raydistance)
                if hit and (normal.dot(invNormal) < 0):
                    hit_dist = Vector((loc[0] - vertLoc[0], loc[1] - vertLoc[1], loc[2] - vertLoc[2])).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                biasVec = tuple([bias*x for x in invNormal])
                rotQuat = forward.rotation_difference(invNormal)

                # offset ray origin with normal bias
                vertPos = (vertLoc[0] + biasVec[0], vertLoc[1] + biasVec[1], vertLoc[2] + biasVec[2])

                for sample in hemiSphere:
                    sample = Vector(sample)
                    sample.rotate(rotQuat)

                    hit, loc, normal, index = obj.ray_cast(vertPos, sample, distance=raydistance)

                    if hit:
                        hitfunction(vert_id, loc, vertPos, dist_list)

        contribution = 1.0/float(raycount)
        forward = Vector((0.0, 0.0, 1.0))

        dist_list = []
        vert_occ_dict = {}
        vert_dict = self.vertex_data_dict(obj)

        if len(vert_dict.keys()) > 0:
            for modifier in obj.modifiers:
                if modifier.type == 'SUBSURF':
                    modifier.show_viewport = False

            # First pass to analyze ray hit distances,
            # then set max ray distance to half of median distance
            ray_caster(obj, 20, vert_dict, dist_hit)
            distance = statistics.median(dist_list) * 0.5

            # Second pass for final results
            ray_caster(obj, raycount, vert_dict, thick_hit, raydistance=distance)

            for modifier in obj.modifiers:
                if modifier.type == 'SUBSURF':
                    modifier.show_viewport = True

            return generate.vert_dict_to_loop_list(obj, vert_occ_dict, 1, 4)
        else:
            return None


    def occlusion_list(self, obj, raycount=500, blend=0.5, dist=10.0, groundplane=False):
        scene = bpy.context.scene
        contribution = 1.0/float(raycount)
        hemiSphere = self.ray_randomizer(raycount)
        mix = max(min(blend, 1.0), 0.0)
        forward = Vector((0.0, 0.0, 1.0))

        edg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(edg)

        vert_occ_dict = {}
        vert_dict = self.vertex_data_dict(obj)

        if len(vert_dict.keys()) > 0:

            if groundplane:
                pivot = utils.find_root_pivot([obj, ])
                pivot = (pivot[0], pivot[1], -0.5)  # pivot[2] - 0.5)
                ground, groundmesh = self.ground_plane(20, pivot)

            for vert_id in vert_dict:
                bias = 0.001
                occValue = 1.0
                scnOccValue = 1.0
                vertLoc = Vector(vert_dict[vert_id][0])
                vertNormal = Vector(vert_dict[vert_id][1])
                vertWorldLoc = Vector(vert_dict[vert_id][2])
                vertWorldNormal = Vector(vert_dict[vert_id][3])

                # Pass 0: Raycast for bias
                hit, loc, normal, index = obj.ray_cast(vertLoc, vertNormal, distance=dist)
                if hit and (normal.dot(vertNormal) > 0):
                    hit_dist = Vector((loc[0] - vertLoc[0], loc[1] - vertLoc[1], loc[2] - vertLoc[2])).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                # Pass 1: Local space occlusion for individual object
                if 0.0 <= mix < 1.0:
                    biasVec = tuple([bias*x for x in vertNormal])
                    rotQuat = forward.rotation_difference(vertNormal)

                    # offset ray origin with normal bias
                    vertPos = (vertLoc[0] + biasVec[0], vertLoc[1] + biasVec[1], vertLoc[2] + biasVec[2])

                    for sample in hemiSphere:
                        sample = Vector(sample)
                        sample.rotate(rotQuat)

                        hit, loc, normal, index = obj_eval.ray_cast(vertPos, sample, distance=dist)

                        if hit:
                            occValue -= contribution

                # Pass 2: Worldspace occlusion for scene
                if 0.0 < mix <= 1.0:
                    biasVec = tuple([bias*x for x in vertWorldNormal])
                    rotQuat = forward.rotation_difference(vertWorldNormal)

                    # offset ray origin with normal bias
                    scnVertPos = (vertWorldLoc[0] + biasVec[0], vertWorldLoc[1] + biasVec[1], vertWorldLoc[2] + biasVec[2])

                    for sample in hemiSphere:
                        sample = Vector(sample)
                        sample.rotate(rotQuat)

                        scnHit, scnLoc, scnNormal, scnIndex, scnObj, ma = scene.ray_cast(edg, scnVertPos, sample, distance=dist)
                        # scene.ray_cast(scene.view_layers[0].depsgraph, scnVertPos, sample, distance=dist)

                        if scnHit:
                            scnOccValue -= contribution

                vert_occ_dict[vert_id] = float((occValue * (1.0 - mix)) + (scnOccValue * mix))

            if groundplane:
                bpy.data.objects.remove(ground, do_unlink=True)
                bpy.data.meshes.remove(groundmesh, do_unlink=True)

            return generate.vert_dict_to_loop_list(obj, vert_occ_dict, 1, 4)

        else:
            return None


    def color_list(self, obj, color):
        count = len(obj.data.attributes[0].data)
        colors = [color[0], color[1], color[2], color[3]] * count

        return colors


    def vertex_id_list(self, obj):
        mesh = obj.data

        count = len(mesh.vertices)
        ids = [None] * count
        mesh.vertices.foreach_get('index', ids)
        return ids


    def empty_list(self, obj, channelcount):
        count = len(obj.data.uv_layers[0].data)
        looplist = [0.0] * count * channelcount

        return looplist


    def vert_dict_to_loop_list(self, obj, vert_dict, dictchannelcount, listchannelcount):
        mesh = obj.data
        loop_list = self.empty_list(obj, listchannelcount)

        if dictchannelcount < listchannelcount:
            if (dictchannelcount == 1) and (listchannelcount == 2):
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, 0.0)
                        loop_list[(0+i*listchannelcount):(listchannelcount+i*listchannelcount)] = [value, value]
                        i += 1
            elif (dictchannelcount == 1) and (listchannelcount == 4):
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, 0.0)
                        loop_list[(0+i*listchannelcount):(listchannelcount+i*listchannelcount)] = [value, value, value, 1.0]
                        i += 1
            elif (dictchannelcount == 3) and (listchannelcount == 4):
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        value = vert_dict.get(vert_idx, [0.0, 0.0, 0.0])
                        loop_list[(0+i*listchannelcount):(listchannelcount+i*listchannelcount)] = [value[0], value[1], value[2], 1.0]
                        i += 1
        else:
            if listchannelcount == 1:
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        loop_list[i] = vert_dict.get(vert_idx, 0.0)
                        i += 1
            else:
                i = 0
                for poly in mesh.polygons:
                    for vert_idx, loop_idx in zip(poly.vertices, poly.loop_indices):
                        loop_list[(0+i*listchannelcount):(listchannelcount+i*listchannelcount)] = vert_dict.get(vert_idx, [0.0] * listchannelcount)
                        i += 1

        return loop_list


    def vertex_data_dict(self, obj):
        mesh = obj.data
        mat = obj.matrix_world
        ids = self.vertex_id_list(obj)

        vertex_dict = {}
        for vert_id in ids:
            vertex_dict[vert_id] = (mesh.vertices[vert_id].co, mesh.vertices[vert_id].normal, mat @ mesh.vertices[vert_id].co, (mat @ mesh.vertices[vert_id].normal - mat @ Vector()).normalized())

        return vertex_dict


    def __del__(self):
        print('SX Tools: Exiting generate')


# ------------------------------------------------------------------------
#    Layer Functions
#    NOTE: Objects must be in OBJECT mode before calling layer functions,
#          use utils.mode_manager() before calling layer functions
#          to set and track correct state
# ------------------------------------------------------------------------
class SXAO_layers(object):
    def __init__(self):
        return None


    def get_colors(self, obj, source):
        sourceColors = obj.data.attributes[source].data
        colors = [None] * len(sourceColors) * 4
        sourceColors.foreach_get('color', colors)
        return colors


    def set_colors(self, obj, target, colors):
        targetColors = obj.data.attributes[target].data
        targetColors.foreach_set('color', colors)


    def __del__(self):
        print('SX Tools: Exiting layers')


# ------------------------------------------------------------------------
#    Tool Actions
# ------------------------------------------------------------------------
class SXAO_tools(object):
    def __init__(self):
        return None


    def apply_tool(self, objs, targetlayer):
        # then = time.perf_counter()
        scene = bpy.context.scene.sxao

        for obj in objs:
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
            if targetlayer not in obj.data.attributes:
                obj.data.attributes.new(name=targetlayer, type='FLOAT_COLOR', domain='CORNER')
                obj.data.attributes.active_color = obj.data.attributes[targetlayer]

            # Get colorbuffer
            if scene.toolmode == 'OCC':
                colors = generate.occlusion_list(obj, scene.occlusionrays, scene.occlusionblend, scene.occlusiondistance, scene.occlusiongroundplane)
            elif scene.toolmode == 'THK':
                colors = generate.thickness_list(obj, scene.occlusionrays)

            if colors is not None:
                layers.set_colors(obj, targetlayer, colors)

        # now = time.perf_counter()
        # print('Apply tool ', scene.toolmode, ' duration: ', now-then, ' seconds')


    def __del__(self):
        print('SX Tools: Exiting tools')

# ------------------------------------------------------------------------
#    Core Functions
# ------------------------------------------------------------------------

# ARMATURE objects pass a more complex validation, return value eliminates duplicates
def selection_validator(self, context):
    selObjs = []
    for obj in context.view_layer.objects.selected:
        if obj.type == 'MESH' and obj.hide_viewport is False:
            selObjs.append(obj)
        elif obj.type == 'ARMATURE':
            all_children = utils.find_children(obj, recursive=True)
            for child in all_children:
                if child.type == 'MESH':
                    selObjs.append(child)

    return list(set(selObjs))


def expand_element(self, context, element):
    if not getattr(context.scene.sxao, element):
        setattr(context.scene.sxao, element, True)


class SXAO_sceneprops(bpy.types.PropertyGroup):
    toolmode: bpy.props.EnumProperty(
        name='Tool Mode',
        description='Select tool',
        items=[
            ('OCC', 'Ambient Occlusion', ''),
            ('THK', 'Mesh Thickness', '')],
        default='OCC',
        update=lambda self, context: expand_element(self, context, 'expandfill'))

    occlusionblend: bpy.props.FloatProperty(
        name='Occlusion Blend',
        description='Blend between self-occlusion and\nthe contribution of all objects in the scene',
        min=0.0,
        max=1.0,
        default=0.5)

    occlusionrays: bpy.props.IntProperty(
        name='Ray Count',
        description='Increase ray count to reduce noise',
        min=1,
        max=5000,
        default=500)

    occlusiondistance: bpy.props.FloatProperty(
        name='Ray Distance',
        description='How far a ray can travel without\nhitting anything before being a miss',
        min=0.0,
        max=100.0,
        default=10.0)

    occlusiongroundplane: bpy.props.BoolProperty(
        name='Ground Plane',
        description='Enable temporary ground plane for occlusion (height -0.5)',
        default=True)

    expandfill: bpy.props.BoolProperty(
        name='Expand Fill',
        default=False)


# ------------------------------------------------------------------------
#    UI Panel
# ------------------------------------------------------------------------
class SXAO_PT_panel(bpy.types.Panel):

    bl_idname = 'SXAO_PT_panel'
    bl_label = 'SX Ambient Occlusion'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SX Ambient Occlusion'


    def draw(self, context):
        objs = selection_validator(self, context)
        layout = self.layout

        if (len(objs) > 0):
            scene = context.scene.sxao

            # Fill Tools --------------------------------------------------------
            box_fill = layout.box()
            row_fill = box_fill.row()
            row_fill.prop(
                scene, 'expandfill',
                icon='TRIA_DOWN' if scene.expandfill else 'TRIA_RIGHT',
                icon_only=True, emboss=False)
            row_fill.prop(scene, 'toolmode', text='')
            row_fill.operator('sxao.applytool', text='Apply')

            if scene.expandfill:
                col_fill = box_fill.column(align=True)
                if scene.toolmode == 'OCC' or scene.toolmode == 'THK':
                    col_fill.prop(scene, 'occlusionrays', slider=True, text='Ray Count')
                if scene.toolmode == 'OCC':
                    col_fill.prop(scene, 'occlusionblend', slider=True, text='Local/Global Mix')
                    col_fill.prop(scene, 'occlusiondistance', slider=True, text='Ray Distance')
                    row_ground = col_fill.row(align=False)
                    row_ground.prop(scene, 'occlusiongroundplane', text='Ground Plane')

                    if scene.occlusionblend == 0:
                        row_ground.enabled = False
        else:
            col = layout.column()
            col.label(text='Select one or multiple meshes to continue')


# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------
class SXAO_OT_applytool(bpy.types.Operator):
    bl_idname = 'sxao.applytool'
    bl_label = 'Apply Tool'
    bl_description = 'Applies the selected mode fill\nto the selected components or objects'
    bl_options = {'UNDO'}


    def invoke(self, context, event):
        objs = selection_validator(self, context)

        scene = bpy.context.scene.sxao
        mode_dict = {'OCC': 'occlusion', 'THK': 'thickness'}

        if len(objs) > 0:
            tools.apply_tool(objs, mode_dict[scene.toolmode])
            bpy.context.space_data.shading.type = 'SOLID'
            bpy.context.space_data.shading.color_type = 'VERTEX'
            bpy.context.space_data.shading.light = 'FLAT'

            for obj in objs:
                obj.data.attributes.active_color = obj.data.attributes[mode_dict[scene.toolmode]]

        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Registration and initialization
# ------------------------------------------------------------------------
sxglobals = SXAO_sxglobals()
utils = SXAO_utils()
generate = SXAO_generate()
layers = SXAO_layers()
tools = SXAO_tools()

classes = (
    SXAO_sceneprops,
    SXAO_PT_panel,
    SXAO_OT_applytool)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Scene.sxao = bpy.props.PointerProperty(type=SXAO_sceneprops)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Object.sxao
    del bpy.types.Object.sxlayers
    del bpy.types.Scene.sxao


if __name__ == '__main__':
    try:
        unregister()
    except:
        pass
    register()
