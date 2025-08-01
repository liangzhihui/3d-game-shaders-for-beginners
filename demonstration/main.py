from numpy import pi as M_PI
from numpy import cos, sin, radians
from direct.showbase.ShowBase import ShowBase

# from direct.showbase import ShowBaseGlobal

from panda3d.core import (
    loadPrcFile,
    NodePath,
    LColor,
    LVecBase4,
    LVecBase2f,
    LVecBase3,
    LVecBase3f,
    PandaNode,
    TransparencyAttrib,
    AmbientLight,
    Shader,
    GraphicsOutput,
    FrameBufferProperties,
    Texture,
    GraphicsPipe,
    WindowProperties,
    CardMaker,
    Camera,
    OrthographicLens,
    LVecBase4f,
    SamplerState,
    BitMask32,
    DirectionalLight,
    LVector3f,
    LPoint3f,
    AnimControlCollection,
    auto_bind,
    PartGroup,
)
from panda3d.physics import (
    ParticleSystem,
    ForceNode,
    PhysicalNode,
    PointParticleFactory,
    SpriteParticleRenderer,
    BaseParticleRenderer,
    PointEmitter,
    BaseParticleEmitter,
    LinearVectorForce,
    LinearJitterForce,
    LinearCylinderVortexForce,
    PhysicsManager,
    ParticleSystemManager,
)


class FramebufferTextureArguments:
    def __init__(self):
        self.window = None
        self.graphicsOutput = None
        self.graphicsEngine = None

        self.bitplane = None
        self.rgbaBits = LVecBase4(0)
        self.clearColor = LColor(0)
        self.aux_rgba = 0
        self.setFloatColor = False
        self.setSrgbColor = False
        self.setRgbColor = False
        self.useScene = False
        self.name = "no name"


class FramebufferTexture:
    def __init__(self):
        self.buffer: GraphicsOutput = None
        self.bufferRegion = None
        self.camera: Camera = None
        self.cameraNP: NodePath = None
        self.shaderNP: NodePath = None


loadPrcFile("panda3d-prc-file.prc")

backgroundColor = [LColor(0.392, 0.537, 0.561, 1), LColor(0.953, 0.733, 0.525, 1)]

TO_RAD = M_PI / 180.0
PI_SHADER_INPUT = LVecBase2f(M_PI, TO_RAD)

GAMMA = 2.2
GAMMA_REC = 1.0 / GAMMA
GAMMA_SHADER_INPUT = LVecBase2f(GAMMA, GAMMA_REC)

BACKGROUND_RENDER_SORT_ORDER = 10
UNSORTED_RENDER_SORT_ORDER = 50

SSAO_SAMPLES = 8
SSAO_NOISE = 4

SHADOW_SIZE = 2048

sunlightColor0 = LVecBase4f(0.612, 0.365, 0.306, 1)
sunlightColor1 = LVecBase4f(0.765, 0.573, 0.400, 1)
moonlightColor0 = LVecBase4f(0.247, 0.384, 0.404, 1)
moonlightColor1 = LVecBase4f(0.392, 0.537, 0.571, 1)
windowLightColor = LVecBase4f(0.765, 0.573, 0.400, 1)

previousViewWorldMat = None
currentViewWorldMat = None


def setTextureToNearestAndClamp(texture):
    texture.set_magfilter(SamplerState.FT_nearest)
    texture.set_minfilter(SamplerState.FT_nearest)
    texture.set_wrap_u(SamplerState.WM_clamp)
    texture.set_wrap_v(SamplerState.WM_clamp)
    texture.set_wrap_w(SamplerState.WM_clamp)


cameraRotatePhiInitial = 67.5095
cameraRotateThetaInitial = 231.721
cameraRotateRadiusInitial = 1100.83
cameraLookAtInitial = LVecBase3(1.00839, 1.20764, 5.85055)
cameraFov = 1.0
cameraNear = 150
cameraFar = 2000
cameraNearFar = LVecBase2f(cameraNear, cameraFar)
cameraRotateRadius = cameraRotateRadiusInitial
cameraRotatePhi = cameraRotatePhiInitial
cameraRotateTheta = cameraRotateThetaInitial
cameraLookAt = cameraLookAtInitial

fogNearInitial = 2.0
fogFarInitial = 90.0
fogNear = fogNearInitial
fogFar = fogFarInitial
fogAdjust = 0.1

foamDepthInitial = LVecBase2f(1.5, 1.5)
foamDepthAdjust = 0.1
foamDepth = foamDepthInitial

mouseThen = LVecBase2f(0.0, 0.0)
mouseNow = mouseThen
mouseWheelDown = False
mouseWheelUp = False

riorInitial = LVecBase2f(1.05, 1.05)
riorAdjust = 0.005
rior = riorInitial

mouseFocusPointInitial = LVecBase2f(0.509167, 0.598)
mouseFocusPoint = mouseFocusPointInitial

sunlightP = 260
animateSunlight = True

soundEnabled = True
soundStarted = False
startSoundAt = 0.5

closedShutters = True

statusAlpha = 1.0
statusColor = LColor(0.9, 0.9, 1.0, statusAlpha)
statusShadowColor = LColor(0.1, 0.1, 0.3, statusAlpha)
statusFadeRate = 2.0
statusText = "Ready"

ssaoEnabled = LVecBase2f(1)
blinnPhongEnabled = LVecBase2f(1)
fresnelEnabled = LVecBase2f(1)
rimLightEnabled = LVecBase2f(1)
refractionEnabled = LVecBase2f(1)
reflectionEnabled = LVecBase2f(1)
fogEnabled = LVecBase2f(1)
outlineEnabled = LVecBase2f(1)
celShadingEnabled = LVecBase2f(1)
normalMapsEnabled = LVecBase2f(1)
bloomEnabled = LVecBase2f(1)
sharpenEnabled = LVecBase2f(1)
depthOfFieldEnabled = LVecBase2f(1)
filmGrainEnabled = LVecBase2f(1)
flowMapsEnabled = LVecBase2f(1)
lookupTableEnabled = LVecBase2f(1)
painterlyEnabled = LVecBase2f(0)
motionBlurEnabled = LVecBase2f(0)
posterizeEnabled = LVecBase2f(0)
pixelizeEnabled = LVecBase2f(0)
chromaticAberrationEnabled = LVecBase2f(1)

rgba8 = LVecBase4(8, 8, 8, 8)
rgba16 = LVecBase4(16, 16, 16, 16)
rgba32 = LVecBase4(32, 32, 32, 32)

base: ShowBase


def calculateCameraPosition(radius, phi, theta, lookAt):
    x = radius * sin(radians(phi)) * cos(radians(theta)) + lookAt[0]
    y = radius * sin(radians(phi)) * sin(radians(theta)) + lookAt[1]
    z = radius * cos(radians(phi)) + lookAt[2]
    return LVecBase3f(x, y, z)


def squashGeometry(environmentNP: NodePath):
    for i in range(4):
        treeCollection = environmentNP.find_all_matches("**/tree" + str(i))
        treesNP = NodePath("treeCollection" + str(i))
        treesNP.reparentTo(environmentNP)
        treeCollection.reparentTo(treesNP)
        treesNP.flatten_strong()


def setUpParticles(render, smokeTexture):
    smokePS = ParticleSystem()
    smokeFN = ForceNode("smoke")
    smokePN = PhysicalNode("smoke")

    smokePS.set_pool_size(75)
    smokePS.set_birth_rate(0.01)
    smokePS.set_litter_size(1)
    smokePS.set_litter_spread(2)
    smokePS.set_system_lifespan(0.0)
    smokePS.set_local_velocity_flag(True)
    smokePS.set_system_grows_older_flag(False)

    smokePPF = PointParticleFactory()
    smokePPF.set_lifespan_base(0.1)
    smokePPF.set_lifespan_spread(3)
    smokePPF.set_mass_base(1)
    smokePPF.set_mass_spread(0)
    smokePPF.set_terminal_velocity_base(400)
    smokePPF.set_terminal_velocity_spread(0)
    smokePS.set_factory(smokePPF)

    smokeSPR = SpriteParticleRenderer()
    smokeSPR.set_alpha_mode(BaseParticleRenderer.PR_ALPHA_OUT)
    smokeSPR.set_user_alpha(1.0)
    smokeSPR.set_texture(smokeTexture)
    smokeSPR.set_color(LColor(1.0, 1.0, 1.0, 1.0))
    smokeSPR.set_x_scale_flag(True)
    smokeSPR.set_y_scale_flag(True)
    smokeSPR.set_anim_angle_flag(True)
    smokeSPR.set_initial_x_scale(0.0000001)
    smokeSPR.set_final_x_scale(0.007)
    smokeSPR.set_initial_y_scale(0.0000001)
    smokeSPR.set_final_y_scale(0.007)
    smokeSPR.set_nonanimated_theta(209.0546)
    smokeSPR.set_alpha_blend_method(BaseParticleRenderer.PP_BLEND_CUBIC)
    smokeSPR.set_alpha_disable(False)
    smokeSPR.get_color_interpolation_manager().add_linear(
        0.0, 1.0, LColor(1.0, 1.0, 1.0, 1.0), LColor(0.039, 0.078, 0.156, 1.0), True
    )
    smokePS.set_renderer(smokeSPR)

    smokePE = PointEmitter()
    smokePE.set_emission_type(BaseParticleEmitter.ET_EXPLICIT)
    smokePE.set_amplitude(0.0)
    smokePE.set_amplitude_spread(1.0)
    smokePE.set_offset_force(LVector3f(0.0, 0.0, 2.0))
    smokePE.set_explicit_launch_vector(LVector3f(0.0, 0.1, 0.0))
    smokePE.set_radiate_origin(LPoint3f(0.0, 0.0, 0.0))
    smokePE.set_location(LPoint3f(0.0, 0.0, 0.0))
    smokePS.set_emitter(smokePE)

    smokeLVF = LinearVectorForce(LVector3f(3.0, -2.0, 0.0), 1.0, False)
    smokeLVF.set_vector_masks(True, True, True)
    smokeLVF.set_active(True)
    smokeFN.add_force(smokeLVF)
    smokePS.add_linear_force(smokeLVF)

    smokeLJF = LinearJitterForce(2.0, False)
    smokeLJF.set_vector_masks(True, True, True)
    smokeLJF.set_active(True)
    smokeFN.add_force(smokeLJF)
    smokePS.add_linear_force(smokeLJF)

    smokeLCVF = LinearCylinderVortexForce(10.0, 1.0, 4.0, 1.0, False)
    smokeLCVF.set_vector_masks(True, True, True)
    smokeLCVF.set_active(True)
    smokeFN.add_force(smokeLCVF)
    smokePS.add_linear_force(smokeLCVF)

    smokePN.insert_physical(0, smokePS)
    smokePS.set_render_parent(smokePN)
    smokeNP = render.attach_new_node(smokePN)
    smokeNP.attach_new_node(smokeFN)

    particleSystemManager.attach_particlesystem(smokePS)
    physicsManager.attach_physical(smokePS)

    smokeNP.set_pos(0.47, 4.5, 8.9)
    smokeNP.set_transparency(TransparencyAttrib.M_dual)
    smokeNP.set_bin("fixed", 0)
    return smokeNP


def generateLights(render: NodePath, showLights: bool):
    ambientLight = AmbientLight("ambientLight")
    ambientLight.setColor(LVecBase4(0.388, 0.356, 0.447, 1.0))
    ambientLightNP = render.attach_new_node(ambientLight)
    render.setLight(ambientLightNP)

    sunlight = DirectionalLight("sunlight")
    sunlight.setColor(sunlightColor1)
    sunlight.setShadowCaster(True, SHADOW_SIZE, SHADOW_SIZE)
    sunlight.getLens().setFilmSize(35, 35)
    sunlight.getLens().setNearFar(5.0, 35.0)
    if showLights:
        sunlight.show_frustum()
    sunlightNP = render.attachNewNode(sunlight)
    sunlightNP.setName("sunlight")
    render.setLight(sunlightNP)


def loadShader(vs, fs):
    shader = Shader.load(
        Shader.SL_GLSL,
        vertex="shaders/vertex/" + vs + ".vert",
        fragment="shaders/fragment/" + fs + ".frag",
    )
    return shader


def generateFramebufferTexture(framebufferTextureArguments):
    window = framebufferTextureArguments.window
    graphicsOutput = framebufferTextureArguments.graphicsOutput
    graphicsEngine = framebufferTextureArguments.graphicsEngine

    rgbaBits = framebufferTextureArguments.rgbaBits
    bitplane = framebufferTextureArguments.bitplane
    aux_rgba = framebufferTextureArguments.aux_rgba
    setFloatColor = framebufferTextureArguments.setFloatColor
    setSrgbColor = framebufferTextureArguments.setSrgbColor
    setRgbColor = framebufferTextureArguments.setRgbColor
    useScene = framebufferTextureArguments.useScene
    name = framebufferTextureArguments.name
    clearColor = framebufferTextureArguments.clearColor

    fbp = FrameBufferProperties()
    fbp.setBackBuffers(0)
    fbp.setRgbaBits(
        int(rgbaBits[0]),
        int(rgbaBits[1]),
        int(rgbaBits[2]),
        int(rgbaBits[3])
    )
    fbp.setAuxRgba(aux_rgba)
    fbp.setFloatColor(setFloatColor)
    fbp.setSrgbColor(setSrgbColor)
    fbp.setRgbColor(setRgbColor)
    fbp.setFloatDepth(True)

    buffer = graphicsEngine.make_output(
        graphicsOutput.getPipe(),
        name + "Buffer",
        BACKGROUND_RENDER_SORT_ORDER - 1,
        fbp,
        WindowProperties.size(0, 0),
        GraphicsPipe.BF_refuse_window
        | GraphicsPipe.BF_resizeable
        | GraphicsPipe.BF_can_bind_every
        | GraphicsPipe.BF_rtt_cumulative
        | GraphicsPipe.BF_size_track_host,
        graphicsOutput.getGsg(),
        graphicsOutput.getHost(),
    )

    buffer.addRenderTexture(Texture(), GraphicsOutput.RTM_bind_or_copy, bitplane)
    buffer.setClearColor(clearColor)

    if useScene:
        cameraNP = base.makeCamera(base.win)
        cameraNP.setPosHpr(base.cam.getPos(), base.cam.getHpr())
        camera = cameraNP.node()
        # dr = camera.getDisplayRegion(0)
        # if dr is not None:
        #     dr.setActive(False)
        camera.setLens(mainCamera.getLens())
    else:
        camera = Camera(name + "Camera")
        lens = OrthographicLens()
        lens.setFilmSize(2, 2)
        lens.setFilmOffset(0, 0)
        lens.setNearFar(-1, 1)
        camera.setLens(lens)
        cameraNP = NodePath(camera)

    bufferRegion = buffer.makeDisplayRegion(0, 1, 0, 1)
    bufferRegion.setCamera(cameraNP)

    shaderNP = NodePath(name + "Shader")

    if not useScene:
        renderNP = NodePath(name + "Render")
        renderNP.setDepthTest(False)
        renderNP.setDepthWrite(False)
        cameraNP.reparentTo(renderNP)
        card = CardMaker(name)
        card.setFrameFullscreenQuad()
        card.setHasUvs(True)
        cardNP = NodePath(card.generate())
        cardNP.reparentTo(renderNP)
        cardNP.setPos(0, 0, 0)
        cardNP.setHpr(0, 0, 0)
        cameraNP.lookAt(cardNP)

    result = FramebufferTexture()
    result.buffer = buffer
    result.bufferRegion = bufferRegion
    result.camera = camera
    result.cameraNP = cameraNP
    result.shaderNP = shaderNP
    return result


def hideBuffer(render2d):
    nodePath = render2d.find("**/texture card")
    if nodePath:
        nodePath.detachNode()


def showBuffer(render2d, statusNP, bufferTexture, alpha):
    hideBuffer(render2d)
    bufferName, buffer, texture = bufferTexture
    nodePath = buffer.getTextureCard()
    nodePath.setTexture(buffer.getTexture(texture))
    nodePath.reparentTo(render2d)
    nodePath.setY(0)
    if alpha:
        nodePath.setTransparency(TransparencyAttrib.M_alpha)
    if statusNP is not None:
        statusNP.reparentTo(render2d)


base = ShowBase()
render = base.render
render2d = base.render2d
loader = base.loader
taskManager = base.taskMgr
particleSystemManager = ParticleSystemManager()
physicsManager = PhysicsManager()

# GraphicsWindow
window = base.win
# GraphicsWindow
graphicsWindow = window
# GraphicsOutput
graphicsOutput = window
# graphicsStateGuardian
graphicsStateGuardian = graphicsOutput.getGsg()
# GraphicsEngine
graphicsEngine = base.graphicsEngine


window.setClearColorActive(True)
window.setClearDepthActive(True)
window.setClearStencilActive(True)
window.setClearColor(backgroundColor[1])
window.setClearDepth(1.0)
window.setClearStencil(0)

cameraNP = base.camera
mainCamera = base.cam.node()
mainLens = mainCamera.get_lens()
mainLens.set_fov(cameraFov)
mainLens.set_near_far(cameraNear, cameraFar)

base.cam.setPos(
    calculateCameraPosition(
        cameraRotateRadius, cameraRotatePhi, cameraRotateTheta, cameraLookAt
    )
)
base.cam.lookAt(cameraLookAt)

blankTexture = loader.loadTexture("images/blank.png")
foamPatternTexture = loader.loadTexture("images/foam-pattern.png")
stillFlowTexture = loader.loadTexture("images/still-flow.png")
upFlowTexture = loader.loadTexture("images/up-flow.png")
colorLookupTableTextureN = loader.loadTexture("images/lookup-table-neutral.png")
colorLookupTableTexture0 = loader.loadTexture("images/lookup-table-0.png")
colorLookupTableTexture1 = loader.loadTexture("images/lookup-table-1.png")
smokeTexture = loader.loadTexture("images/smoke.png")
colorNoiseTexture = loader.loadTexture("images/color-noise.png")

sceneRootPN = PandaNode("sceneRoot")
sceneRootNP = NodePath(sceneRootPN)
sceneRootNP.reparentTo(render)

environmentNP: NodePath = loader.loadModel("eggs/mill-scene/mill-scene.bam")
environmentNP.reparentTo(sceneRootNP)
shuttersNP: NodePath = loader.loadModel("eggs/mill-scene/shutters.bam")
shuttersNP.reparentTo(sceneRootNP)
weatherVaneNP: NodePath = loader.loadModel("eggs/mill-scene/weather-vane.bam")
weatherVaneNP.reparentTo(sceneRootNP)
bannerNP: NodePath = loader.loadModel("eggs/mill-scene/banner.bam")
bannerNP.reparentTo(sceneRootNP)

wheelNP: NodePath = environmentNP.find("**/wheel-lp")
waterNP: NodePath = environmentNP.find("**/water-lp")

shuttersAnimationCollection = AnimControlCollection()
weatherVaneAnimationCollection = AnimControlCollection()
bannerAnimationCollection = AnimControlCollection()
auto_bind(
    shuttersNP.node(),
    shuttersAnimationCollection,
    PartGroup.HMF_ok_wrong_root_name
    | PartGroup.HMF_ok_part_extra
    | PartGroup.HMF_ok_anim_extra,
)
auto_bind(
    weatherVaneNP.node(),
    weatherVaneAnimationCollection,
    PartGroup.HMF_ok_wrong_root_name
    | PartGroup.HMF_ok_part_extra
    | PartGroup.HMF_ok_anim_extra,
)
auto_bind(
    bannerNP.node(),
    bannerAnimationCollection,
    PartGroup.HMF_ok_wrong_root_name
    | PartGroup.HMF_ok_part_extra
    | PartGroup.HMF_ok_anim_extra,
)

squashGeometry(environmentNP)

smokeNP = setUpParticles(render, smokeTexture)

waterNP.setTransparency(TransparencyAttrib.M_dual)
waterNP.setBin("fixed", 0)

generateLights(render, False)

discardShader = loadShader("discard", "discard")
baseShader = loadShader("base", "base")
geometryBufferShader0 = loadShader("base", "geometry-buffer-0")
geometryBufferShader1 = loadShader("base", "geometry-buffer-1")
geometryBufferShader2 = loadShader("base", "geometry-buffer-2")
foamShader = loadShader("basic", "foam")
fogShader = loadShader("basic", "fog")
boxBlurShader = loadShader("basic", "box-blur")
motionBlurShader = loadShader("basic", "motion-blur")
kuwaharaFilterShader = loadShader("basic", "kuwahara-filter")
dilationShader = loadShader("basic", "dilation")
sharpenShader = loadShader("basic", "sharpen")
outlineShader = loadShader("basic", "outline")
bloomShader = loadShader("basic", "bloom")
ssaoShader = loadShader("basic", "ssao")
screenSpaceRefractionShader = loadShader("basic", "screen-space-refraction")
screenSpaceReflectionShader = loadShader("basic", "screen-space-reflection")
refractionShader = loadShader("basic", "refraction")
reflectionColorShader = loadShader("basic", "reflection-color")
reflectionShader = loadShader("basic", "reflection")
baseCombineShader = loadShader("basic", "base-combine")
sceneCombineShader = loadShader("basic", "scene-combine")
depthOfFieldShader = loadShader("basic", "depth-of-field")
posterizeShader = loadShader("basic", "posterize")
pixelizeShader = loadShader("basic", "pixelize")
filmGrainShader = loadShader("basic", "film-grain")
lookupTableShader = loadShader("basic", "lookup-table")
gammaCorrectionShader = loadShader("basic", "gamma-correction")
chromaticAberrationShader = loadShader("basic", "chromatic-aberration")

mainCameraNP = NodePath("mainCamera")
mainCameraNP.setShader(discardShader)
mainCamera.set_initial_state(mainCameraNP.getState())

isWaterNP = NodePath("isWater")
isWaterNP.setShaderInput("isWater", (1.0, 1.0))
isWaterNP.setShaderInput("flowTexture", upFlowTexture)
isWaterNP.setShaderInput("foamPatternTexture", foamPatternTexture)

isSmokeNP = NodePath("isSmoke")
isSmokeNP.setShaderInput("isSmoke", (1.0, 1.0))
isSmokeNP.setShaderInput("isParticle", (1.0, 1.0))

currentViewWorldMat = cameraNP.getTransform(render).getMat()

framebufferTextureArguments = FramebufferTextureArguments()
framebufferTextureArguments.window = window
framebufferTextureArguments.graphicsOutput = graphicsOutput
framebufferTextureArguments.graphicsEngine = graphicsEngine

framebufferTextureArguments.bitplane = GraphicsOutput.RTP_color
framebufferTextureArguments.rgbaBits = rgba32
framebufferTextureArguments.clearColor = LColor(0, 0, 0, 0)
framebufferTextureArguments.aux_rgba = 1
framebufferTextureArguments.setFloatColor = True
framebufferTextureArguments.setSrgbColor = False
framebufferTextureArguments.setRgbColor = True
framebufferTextureArguments.useScene = True
framebufferTextureArguments.name = "geometry0"

geometryFrameBufferTexture0 = generateFramebufferTexture(framebufferTextureArguments)
geometryBuffer0 = geometryFrameBufferTexture0.buffer
geometryCamera0 = geometryFrameBufferTexture0.camera
geometryNP0: NodePath = geometryFrameBufferTexture0.shaderNP
geometryBuffer0.addRenderTexture(
    Texture(), GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_aux_rgba_0
)
geometryBuffer0.setClearActive(3, True)
geometryBuffer0.setClearValue(3, framebufferTextureArguments.clearColor)
geometryNP0.setShader(geometryBufferShader0)
geometryNP0.setShaderInput("normalMapsEnabled", normalMapsEnabled)
geometryCamera0.setInitialState(geometryNP0.getState())
geometryCamera0.setCameraMask(BitMask32.bit(1))
positionTexture0 = geometryBuffer0.getTexture(0)
normalTexture0 = geometryBuffer0.getTexture(1)
geometryCameraLens0 = geometryCamera0.getLens()
waterNP.hide(BitMask32.bit(1))
smokeNP.hide(BitMask32.bit(1))

showBufferIndex = 0

bufferArray = [
    ("Positions 0", geometryBuffer0, 0),
    ("Normals 0", geometryBuffer0, 1)
]
showBufferIndex = len(bufferArray) - 1
showBuffer(render2d, None, bufferArray[showBufferIndex], False)

shuttersAnimationCollection.play("close-shutters")
weatherVaneAnimationCollection.loop("weather-vane-shake", True)
bannerAnimationCollection.loop("banner-swing", True)


if __name__ == "__main__":
    # render.ls()
    base.run()
