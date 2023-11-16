bl_info = {
    'name': 'SX Ambient Occlusion',
    'author': 'Jani Kahrama / Secret Exit Ltd.',
    'version': (1, 4, 3),
    'blender': (3, 5, 0),
    'location': 'View3D',
    'description': 'Vertex Ambient Occlusion Tool',
    'doc_url': 'https://www.notion.so/SX-Tools-for-Blender-Documentation-9ad98e239f224624bf98246822a671a6',
    'tracker_url': 'https://github.com/FrandSX/sxtools-blender/issues',
    'category': 'Development',
}


import bpy
import random
import time
import math
import bmesh
import statistics
from mathutils import Vector


version, _, _ = bpy.app.version

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
        random.seed(42)

        for i in range(count):
            u1 = random.random()
            u2 = random.random()
            r = math.sqrt(u1)
            theta = 2*math.pi*u2

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = math.sqrt(max(0, 1 - u1))

            ray = Vector((x, y, z))
            up_vector = Vector((0, 0, 1))
            
            dot_product = ray.dot(up_vector)
            hemiSphere[i] = (ray, dot_product)

        sorted_hemiSphere = sorted(hemiSphere, key=lambda x: x[1], reverse=True)
        return sorted_hemiSphere


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


    def create_occlusion_network(self, raycount):
        def ray_randomizer(count):
            hemiSphere = [None] * count
            random.seed(42)

            for i in range(count):
                r = math.sqrt(random.random())
                theta = 2 * math.pi * random.random()
                hemiSphere[i] = (r, theta, 0)

            sorted_hemiSphere = sorted(hemiSphere, key=lambda x: x[1], reverse=True)
            return sorted_hemiSphere


        def connect_nodes(output, input):
            nodetree.links.new(output, input)


        nodetree = bpy.data.node_groups.new(type='GeometryNodeTree', name='sx_ao')

        # expose bias and ground plane inputs
        group_in = nodetree.nodes.new(type='NodeGroupInput')
        group_in.name = 'group_input'
        group_in.location = (-1000, 0)

        if version == 4:
            geometry = nodetree.interface.new_socket(in_out='INPUT', name='Geometry', socket_type='NodeSocketGeometry')
            ray_loops = nodetree.interface.new_socket(in_out='INPUT', name='Raycount', socket_type='NodeSocketInt')
            bias = nodetree.interface.new_socket(in_out='INPUT', name='Ray Bias', socket_type='NodeSocketFloat')
            ground_plane = nodetree.interface.new_socket(in_out='INPUT', name='Ground Plane', socket_type='NodeSocketBool')
            ground_offset = nodetree.interface.new_socket(in_out='INPUT', name='Ground Plane Offset', socket_type='NodeSocketFloat')
            geometry_out = nodetree.interface.new_socket(in_out='OUTPUT', name='Geometry', socket_type='NodeSocketGeometry')
            color_out = nodetree.interface.new_socket(in_out='OUTPUT', name='Color Output', socket_type='NodeSocketColor')
        else:
            geometry = nodetree.inputs.new('NodeSocketGeometry', 'Geometry')
            ray_loops = nodetree.inputs.new('NodeSocketInt', 'Raycount')
            bias = nodetree.inputs.new('NodeSocketFloat', 'Ray Bias')
            ground_plane = nodetree.inputs.new('NodeSocketBool', 'Ground Plane')
            ground_offset = nodetree.inputs.new('NodeSocketFloat', 'Ground Plane Offset')
            geometry_out = nodetree.outputs.new('NodeSocketGeometry', 'Geometry')
            color_out = nodetree.outputs.new('NodeSocketColor', 'Color Output')

        bias.min_value = 0
        bias.max_value = 1
        bias.default_value = 0.001
        ground_plane.default_value = False
        ground_offset.default_value = 0

        # expose group color output
        group_out = nodetree.nodes.new(type='NodeGroupOutput')
        group_out.name = 'group_output'
        group_out.location = (2000, 0)

        color_out.attribute_domain = 'POINT'
        color_out.default_attribute_name = 'occlusion'
        color_out.default_value = (1, 1, 1, 1)

        # vertex inputs
        index = nodetree.nodes.new(type='GeometryNodeInputIndex')
        index.name = 'index'
        index.location = (-1000, 200)
        index.hide = True

        normal = nodetree.nodes.new(type='GeometryNodeInputNormal')
        normal.name = 'normal'
        normal.location = (-1000, 150)
        normal.hide = True

        position = nodetree.nodes.new(type='GeometryNodeInputPosition')
        position.name = 'position'
        position.location = (-1000, 100)
        position.hide = True

        eval_normal = nodetree.nodes.new(type='GeometryNodeFieldAtIndex')
        eval_normal.name = 'eval_normal'
        eval_normal.location = (-800, 200)
        eval_normal.data_type = 'FLOAT_VECTOR'
        eval_normal.domain = 'POINT'
        eval_normal.hide = True

        eval_position = nodetree.nodes.new(type='GeometryNodeFieldAtIndex')
        eval_position.name = 'eval_position'
        eval_position.location = (-800, 150)
        eval_position.data_type = 'FLOAT_VECTOR'
        eval_position.domain = 'POINT'
        eval_position.hide = True

        bias_normal = nodetree.nodes.new(type='ShaderNodeVectorMath')
        bias_normal.name = 'bias_normal'
        bias_normal.location = (-600, 200)
        bias_normal.operation = 'MULTIPLY'
        bias_normal.hide = True

        bias_pos = nodetree.nodes.new(type='ShaderNodeVectorMath')
        bias_pos.name = 'bias_pos'
        bias_pos.location = (-400, 200)
        bias_pos.operation = 'ADD'
        bias_pos.hide = True

        connect_nodes(group_in.outputs['Geometry'], group_out.inputs['Geometry'])

        connect_nodes(index.outputs['Index'], eval_normal.inputs['Index'])
        connect_nodes(index.outputs['Index'], eval_position.inputs['Index'])
        connect_nodes(normal.outputs['Normal'], eval_normal.inputs[3])
        connect_nodes(position.outputs['Position'], eval_position.inputs[3])

        connect_nodes(eval_normal.outputs[2], bias_normal.inputs[0])
        connect_nodes(group_in.outputs['Ray Bias'], bias_normal.inputs[1])

        connect_nodes(bias_normal.outputs[0], bias_pos.inputs[0])
        connect_nodes(eval_position.outputs[2], bias_pos.inputs[1])


        # optional ground plane
        bbx = nodetree.nodes.new(type='GeometryNodeBoundBox')
        bbx.name = 'bbx'
        bbx.location = (-800, -200)
        bbx.hide = True

        bbx_min_separate = nodetree.nodes.new(type='ShaderNodeSeparateXYZ')
        bbx_min_separate.name = 'bbx_min_separate'
        bbx_min_separate.location = (-600, -200)
        bbx_min_separate.hide = True

        ground_bias = nodetree.nodes.new(type='ShaderNodeMath')
        ground_bias.name = 'ground_bias'
        ground_bias.location = (-400, -150)
        ground_bias.operation = 'SUBTRACT'
        ground_bias.inputs[1].default_value = 0.001
        ground_bias.hide = True

        ground_offset_add = nodetree.nodes.new(type='ShaderNodeMath')
        ground_offset_add.name = 'add_hits'
        ground_offset_add.location = (-400, -200)
        ground_offset_add.operation = 'ADD'
        ground_offset_add.inputs[1].default_value = 0
        ground_offset_add.hide = True

        bbx_min_combine = nodetree.nodes.new(type='ShaderNodeCombineXYZ')
        bbx_min_combine.name = 'bbx_min_combine'
        bbx_min_combine.location = (-200, -200)
        bbx_min_combine.hide = True

        bbx_multiply = nodetree.nodes.new(type='ShaderNodeVectorMath')
        bbx_multiply.name = 'bbx_multiply'
        bbx_multiply.location = (-600, -250)
        bbx_multiply.operation = 'MULTIPLY'
        bbx_multiply.inputs[1].default_value = (10, 10, 10)
        bbx_multiply.hide = True

        ground_grid = nodetree.nodes.new(type='GeometryNodeMeshGrid')
        ground_grid.name = 'ground_grid'
        ground_grid.location = (-200, -150)
        ground_grid.inputs[2].default_value = 2
        ground_grid.inputs[3].default_value = 2
        ground_grid.hide = True

        ground_transform = nodetree.nodes.new(type='GeometryNodeTransform')
        ground_transform.name = 'ground_transform'
        ground_transform.location = (0, -200)
        ground_transform.hide = True     

        join = nodetree.nodes.new(type='GeometryNodeJoinGeometry')
        join.name = 'join'
        join.location = (200, -150)
        join.hide = True

        ground_switch = nodetree.nodes.new(type='GeometryNodeSwitch')
        ground_switch.name = 'ground_switch'
        ground_switch.location = (400, -100)
        ground_switch.input_type = 'GEOMETRY'
        ground_switch.hide = True

        connect_nodes(group_in.outputs['Geometry'], bbx.inputs['Geometry'])
        connect_nodes(bbx.outputs['Min'], bbx_min_separate.inputs[0])
        connect_nodes(group_in.outputs['Ground Plane Offset'], ground_bias.inputs[0])
        connect_nodes(ground_bias.outputs[0], ground_offset_add.inputs[0])
        connect_nodes(bbx_min_separate.outputs['Z'], ground_offset_add.inputs[1])
        connect_nodes(ground_offset_add.outputs[0], bbx_min_combine.inputs['Z'])
        connect_nodes(bbx.outputs['Max'], bbx_multiply.inputs[0])
        connect_nodes(bbx_min_combine.outputs[0], ground_transform.inputs['Translation'])
        connect_nodes(bbx_multiply.outputs[0], ground_transform.inputs['Scale'])
        connect_nodes(ground_grid.outputs['Mesh'], ground_transform.inputs['Geometry'])
        connect_nodes(group_in.outputs['Geometry'], join.inputs[0])
        connect_nodes(ground_transform.outputs['Geometry'], join.inputs[0])
        connect_nodes(group_in.outputs['Ground Plane'], ground_switch.inputs[1])
        connect_nodes(group_in.outputs['Geometry'], ground_switch.inputs[14])
        connect_nodes(join.outputs['Geometry'], ground_switch.inputs[15])

        # create raycasts with a loop
        hemisphere = ray_randomizer(raycount)
        previous = None
        for i in range(raycount):
            random_rot = nodetree.nodes.new(type='ShaderNodeVectorRotate')
            random_rot.name = 'random_rot'
            random_rot.location = (600, i * 100 + 400)
            random_rot.rotation_type = 'EULER_XYZ'
            random_rot.inputs[4].default_value = hemisphere[i]
            random_rot.hide =True

            raycast = nodetree.nodes.new(type='GeometryNodeRaycast')
            raycast.name = 'raycast'
            raycast.location = (800, i * 100 + 400)
            if version == 4:
                raycast.inputs[9].default_value = 10
            else:
                raycast.inputs[8].default_value = 10
            raycast.hide = True

            if previous is not None:
                add_hits = nodetree.nodes.new(type='ShaderNodeMath')
                add_hits.name = 'add_hits'
                add_hits.location = (1000, i * 100 + 400)
                add_hits.operation = 'ADD'
                add_hits.hide = True
                connect_nodes(raycast.outputs[0], add_hits.inputs[0])
                connect_nodes(previous.outputs[0], add_hits.inputs[1])
                previous = add_hits
            else:
                previous = raycast

            connect_nodes(eval_normal.outputs[2], random_rot.inputs['Vector'])
            connect_nodes(bias_pos.outputs[0], random_rot.inputs['Center'])
            connect_nodes(ground_switch.outputs[6], raycast.inputs['Target Geometry'])
            connect_nodes(random_rot.outputs[0], raycast.inputs['Ray Direction'])
            connect_nodes(bias_pos.outputs[0], raycast.inputs['Source Position'])


        # normalize hit results
        div_hits = nodetree.nodes.new(type='ShaderNodeMath')
        div_hits.name = 'add_hits'
        div_hits.location = (1400, 300)
        div_hits.operation = 'DIVIDE'

        connect_nodes(group_in.outputs['Raycount'], div_hits.inputs[1])

        color_mix = nodetree.nodes.new(type='ShaderNodeMix')
        color_mix.name = 'color_mix'
        color_mix.data_type = 'RGBA'
        color_mix.location = (1600, 300)
        color_mix.inputs[6].default_value = (1, 1, 1, 1)
        color_mix.inputs[7].default_value = (0, 0, 0, 1)

        connect_nodes(previous.outputs[0], div_hits.inputs[0])
        connect_nodes(div_hits.outputs[0], color_mix.inputs[0])
        connect_nodes(color_mix.outputs[2], group_out.inputs['Color Output'])


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

                for (sample, _) in hemiSphere:
                    hit, loc, normal, _ = obj.ray_cast(vertPos, rotQuat @ Vector(sample), distance=raydistance)

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


    def occlusion_list(self, obj, raycount=250, blend=0.5, dist=10.0, groundplane=False, masklayer=None):
        # start_time = time.time()

        scene = bpy.context.scene
        contribution = 1.0/float(raycount)
        hemiSphere = self.ray_randomizer(raycount)
        mix = max(min(blend, 1.0), 0.0)
        forward = Vector((0.0, 0.0, 1.0))

        edg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(edg)

        vert_occ_dict = {}
        vert_dict = self.vertex_data_dict(obj, dots=True)

        if len(vert_dict.keys()) > 0:

            if groundplane:
                pivot = utils.find_root_pivot([obj, ])
                pivot = (pivot[0], pivot[1], pivot[2] - 0.5)
                size = max(obj.dimensions) * 10
                ground, groundmesh = self.ground_plane(size, pivot)

            for vert_id in vert_dict:
                bias = 0.001
                occValue = 1.0
                scnOccValue = 1.0
                vertLoc = Vector(vert_dict[vert_id][0])
                vertNormal = Vector(vert_dict[vert_id][1])
                vertWorldLoc = Vector(vert_dict[vert_id][2])
                vertWorldNormal = Vector(vert_dict[vert_id][3])
                min_dot = vert_dict[vert_id][4]

                # Pass 0: Raycast for bias
                hit, loc, normal, _ = obj.ray_cast(vertLoc, vertNormal, distance=dist)
                if hit and (normal.dot(vertNormal) > 0):
                    hit_dist = (loc - vertLoc).length
                    if hit_dist < 0.5:
                        bias += hit_dist

                # Pass 1: Mark hits for rays that are inside the mesh
                first_hit_index = raycount
                for i, (_, dot) in enumerate(hemiSphere):
                    if dot < min_dot:
                        first_hit_index = i
                        break

                valid_rays = [ray for ray, _ in hemiSphere[:first_hit_index]]
                occValue -= contribution * (raycount - first_hit_index)

                # Store Pass 2 valid ray hits
                pass2_hits = [False] * len(valid_rays)

                # Pass 2: Local space occlusion for individual object
                if 0.0 <= mix < 1.0:
                    rotQuat = forward.rotation_difference(vertNormal)

                    # offset ray origin with normal bias
                    vertPos = vertLoc + (bias * vertNormal)

                    # for every object ray hit, subtract a fraction from the vertex brightness
                    for i, ray in enumerate(valid_rays):
                        hit = obj_eval.ray_cast(vertPos, rotQuat @ Vector(ray), distance=dist)[0]
                        occValue -= contribution * hit
                        pass2_hits[i] = hit

                # Pass 3: Worldspace occlusion for scene
                if 0.0 < mix <= 1.0:
                    rotQuat = forward.rotation_difference(vertWorldNormal)

                    # offset ray origin with normal bias
                    scnVertPos = vertWorldLoc + (bias * vertWorldNormal)

                    # Include previous pass results
                    scnOccValue = occValue

                    # Fire rays only for samples that had not hit in Pass 2
                    for i, ray in enumerate(valid_rays):
                        if not pass2_hits[i]:
                            hit = scene.ray_cast(edg, scnVertPos, rotQuat @ Vector(ray), distance=dist)[0]
                            scnOccValue -= contribution * hit

                vert_occ_dict[vert_id] = float((occValue * (1.0 - mix)) + (scnOccValue * mix))

            if groundplane:
                bpy.data.objects.remove(ground, do_unlink=True)
                bpy.data.meshes.remove(groundmesh, do_unlink=True)

            # end_time = time.time()  # Stop the timer
            # print("SX Tools: AO rendered in {:.4f} seconds".format(end_time - start_time)) 

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


    def vertex_data_dict(self, obj, dots=False):

        def add_to_dict(vert_id):
            min_dot = None
            if dots:
                dot_list = []
                vert = bm.verts[vert_id]
                num_connected = len(vert.link_edges)
                if num_connected > 0:
                    for edge in vert.link_edges:
                        dot_list.append((vert.normal.normalized()).dot((edge.other_vert(vert).co - vert.co).normalized()))
                min_dot = min(dot_list)

            vertex_dict[vert_id] = (
                mesh.vertices[vert_id].co,
                mesh.vertices[vert_id].normal,
                mat @ mesh.vertices[vert_id].co,
                (mat @ mesh.vertices[vert_id].normal - mat @ Vector()).normalized(),
                min_dot
            )

        mesh = obj.data
        mat = obj.matrix_world
        ids = self.vertex_id_list(obj)

        if dots:
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.normal_update()
            bmesh.types.BMVertSeq.ensure_lookup_table(bm.verts)

        vertex_dict = {}
        for vert_id in ids:
            add_to_dict(vert_id)

        if dots:
            bm.free()

        return vertex_dict


    def __del__(self):
        print('SX Tools: Exiting generate')


# ------------------------------------------------------------------------
#    Layer Functions
#    NOTE: Objects must be in OBJECT mode before calling layer functions
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


def toggle_sxao(self, context):
    objs = selection_validator(self, context)
    for obj in objs:
        if 'occlusion' not in obj.data.attributes:
            obj.data.attributes.new(name='occlusion', type='FLOAT_COLOR', domain='CORNER')
            obj.data.attributes.active_color = obj.data.attributes['occlusion']

        if 'sxAO' not in obj.modifiers:
            if 'sx_ao' not in bpy.data.node_groups:
                generate.create_occlusion_network(bpy.context.scene.sxao.occlusionrays)

            ao = obj.modifiers.new(type='NODES', name='sxAO')
            ao.node_group = bpy.data.node_groups['sx_ao']

        if 'sxAO' in obj.modifiers:
            if version == 4:
                socket_name_0 = 'Socket_1'
                socket_name_1 = 'Socket_3'
                socket_name_2 = 'Socket_4'
            else:
                socket_name_0 = 'Input_4'
                socket_name_1 = 'Input_2'
                socket_name_2 = 'Input_3'

            if bpy.context.scene.sxao.occlusionnodes and (obj.modifiers["sxAO"][socket_name_0] != bpy.context.scene.sxao.occlusionrays):
                bpy.data.node_groups.remove(bpy.data.node_groups['sx_ao'], do_unlink=True)
                generate.create_occlusion_network(bpy.context.scene.sxao.occlusionrays)
                obj.modifiers['sxAO'].node_group = bpy.data.node_groups['sx_ao']

            obj.modifiers['sxAO'].show_viewport = bpy.context.scene.sxao.occlusionnodes
            obj.modifiers["sxAO"][socket_name_1] = bpy.context.scene.sxao.occlusiongroundplane
            obj.modifiers["sxAO"][socket_name_2] = bpy.context.scene.sxao.occlusiongroundplaneoffset
            obj.modifiers["sxAO"][socket_name_0] = bpy.context.scene.sxao.occlusionrays

    if bpy.context.scene.sxao.occlusionnodes:
        bpy.context.space_data.shading.color_type = 'VERTEX'


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
        max=1000,
        default=100,
        update=toggle_sxao)

    occlusiondistance: bpy.props.FloatProperty(
        name='Ray Distance',
        description='How far a ray can travel without\nhitting anything before being a miss',
        min=0.0,
        max=100.0,
        default=10.0)

    occlusiongroundplane: bpy.props.BoolProperty(
        name='Ground Plane',
        description='Enable temporary ground plane for occlusion (height -0.5)',
        default=True,
        update=toggle_sxao)

    occlusionnodes: bpy.props.BoolProperty(
        name='Geonode AO',
        description='Enable experimental geometry nodes based AO',
        default=False,
        update=toggle_sxao)

    occlusiongroundplaneoffset: bpy.props.FloatProperty(
        name='Ground Offset',
        description='Set node AO ground level. Default is bottom of object bounding box.',
        default=0.0,
        update=toggle_sxao)

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
                    if not scene.occlusionnodes:
                        col_fill.prop(scene, 'occlusionblend', slider=True, text='Local/Global Mix')
                        col_fill.prop(scene, 'occlusiondistance', slider=True, text='Ray Distance')
                    if scene.occlusionnodes and scene.occlusiongroundplane:
                        col_fill.prop(scene, 'occlusiongroundplaneoffset')

                    row_ground = col_fill.row(align=False)
                    row_ground.prop(scene, 'occlusiongroundplane', text='Ground Plane')

                    if scene.occlusionblend == 0:
                        row_ground.enabled = False

                    col_fill.prop(scene, 'occlusionnodes')

                        

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
