import time
from numpy import pi as M_PI
from numpy import cos, sin, radians
from numpy.random import rand
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

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
    TextNode,
    FontPool,
    AudioManager,
    ButtonRegistry,
    MouseButton,
    AudioSound,
    Spotlight,
    PerspectiveLens,
    PTA_LVecBase3f,
    LMatrix4,
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
    LinearEulerIntegrator,
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


def makeEnableVec(t: int):
    t = 1 if t >= 1 else 0
    return LVecBase2f(t, t)


def toggleEnableVec(vec: LVecBase2f):
    t = vec[0]
    t = 0 if t >= 1 else 1
    vec[0] = t
    vec[1] = t
    return vec


def setTextureToNearestAndClamp(texture):
    texture.set_magfilter(SamplerState.FT_nearest)
    texture.set_minfilter(SamplerState.FT_nearest)
    texture.set_wrap_u(SamplerState.WM_clamp)
    texture.set_wrap_v(SamplerState.WM_clamp)
    texture.set_wrap_w(SamplerState.WM_clamp)


def mixColor(a, b, factor):
    return a * (1 - factor) + b * factor


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
foamDepth = LVecBase2f(1.5, 1.5)

mouseThen = LVecBase2f(0.0, 0.0)
mouseNow = LVecBase2f(0.0, 0.0)
mouseWheelDown = False
mouseWheelUp = False

riorInitial = LVecBase2f(1.05, 1.05)
riorAdjust = 0.005
rior = LVecBase2f(1.05, 1.05)

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


def calculateCameraLookAt(upDownAdjust, leftRightAdjust, phi, theta, lookAt):
    lookAt[0] += upDownAdjust * sin(radians(-theta - 90)) * cos(radians(phi))
    lookAt[1] += upDownAdjust * cos(radians(-theta - 90)) * cos(radians(phi))
    lookAt[2] -= -upDownAdjust * sin(radians(phi))

    lookAt[0] += leftRightAdjust * sin(radians(-theta))
    lookAt[1] += leftRightAdjust * cos(radians(-theta))
    return lookAt


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
    smokeNP = render.attachNewNode(smokePN)
    smokeNP.attachNewNode(smokeFN)

    particleSystemManager.attach_particlesystem(smokePS)
    physicsManager.attach_physical(smokePS)

    smokeNP.setPos(0.47, 4.5, 8.9)
    smokeNP.setTransparency(TransparencyAttrib.M_dual)
    smokeNP.setBin("fixed", 0)
    return smokeNP


def generateLights(render: NodePath, showLights: bool):
    ambientLight = AmbientLight("ambientLight")
    ambientLight.setColor(LVecBase4(0.388, 0.356, 0.447, 1.0))
    ambientLightNP = render.attachNewNode(ambientLight)
    render.setLight(ambientLightNP)

    sunlight = DirectionalLight("sunlight")
    sunlight.setColor(sunlightColor1)
    sunlight.setShadowCaster(True, SHADOW_SIZE, SHADOW_SIZE)
    sunlight.getLens().setFilmSize(35, 35)
    sunlight.getLens().setNearFar(5.0, 35.0)
    if showLights:
        sunlight.showFrustum()
    sunlightNP = render.attachNewNode(sunlight)
    sunlightNP.setName("sunlight")
    render.setLight(sunlightNP)

    moonlight = DirectionalLight("moonlight")
    moonlight.setColor(moonlightColor1)
    moonlight.setShadowCaster(True, SHADOW_SIZE, SHADOW_SIZE)
    moonlight.getLens().setFilmSize(35, 35)
    moonlight.getLens().setNearFar(5.0, 35.0)
    if showLights:
        moonlight.showFrustum()
    moonLightNP = render.attachNewNode(moonlight)
    moonLightNP.setName("moonlight")
    render.setLight(moonLightNP)

    sunlightPivotNP = NodePath("sunlightPivot")
    sunlightPivotNP.reparentTo(render)
    sunlightPivotNP.setPos(0, 0.5, 15.0)
    sunlightNP.reparentTo(sunlightPivotNP)
    sunlightNP.setPos(0, -17.5, 0)
    sunlightPivotNP.setHpr(135, 340, 0)

    moonLightPivotNP = NodePath("moonlightPivot")
    moonLightPivotNP.reparentTo(render)
    moonLightPivotNP.setPos(0, 0.5, 15.0)
    moonLightNP.reparentTo(moonLightPivotNP)
    moonLightNP.setPos(0, -17.5, 0)
    moonLightPivotNP.setHpr(135, 160, 0)

    generateWindowLight("windowLight", render, LVecBase3(1.5, 2.49, 7.9), showLights)
    generateWindowLight("windowLight1", render, LVecBase3(3.5, 2.49, 7.9), showLights)
    generateWindowLight("windowLight2", render, LVecBase3(3.5, 1.49, 4.5), showLights)


def generateWindowLight(name: str, render: NodePath, position: LVecBase3, show: bool):
    windowLight = Spotlight(name)
    windowLight.setColor(windowLightColor)
    windowLight.setExponent(5)
    windowLight.setAttenuation((1, 0.008, 0))
    windowLight.setMaxDistance(37)

    windowLightLens = PerspectiveLens()
    windowLightLens.setNearFar(0.5, 12)
    windowLightLens.setFov(140)
    windowLight.setLens(windowLightLens)

    if show:
        windowLight.showFrustum()

    windowLightNP = render.attachNewNode(windowLight)
    windowLightNP.setName(name)
    windowLightNP.setPos(position)
    windowLightNP.setHpr(180, 0, 0)
    render.setLight(windowLightNP)


def animateLights(
    render: NodePath,
    shuttersAnimationCollection,
    delta,
    speed,
    middayDown,
    midnightDown,
):
    global closedShutters

    def clamp(a, mn, mx):
        if a > mx:
            a = mx
        if a < mn:
            a = mn
        return a

    sunlightPivotNP = render.find("**/sunlightPivot")
    moonlightPivotNP = render.find("**/moonlightPivot")
    sunlightNP = render.find("**/sunlight")
    moonlightNP = render.find("**/moonlight")
    sunlight = sunlightNP.node()
    moonlight = moonlightNP.node()

    p = sunlightPivotNP.getP()
    p += speed * delta
    if p > 360:
        p = 0
    if p < 0:
        p = 360

    if middayDown:
        p = 270
    elif midnightDown:
        p = 90

    sunlightPivotNP.setP(p)
    moonlightPivotNP.setP(p - 180)

    mixFactor = 1.0 - (sin(radians(p)) / 2.0 + 0.5)

    sunlightColor = mixColor(sunlightColor0, sunlightColor1, mixFactor)
    moonlightColor = mixColor(moonlightColor1, sunlightColor0, mixFactor)
    lightColor = mixColor(moonlightColor, sunlightColor, mixFactor)

    dayTimeLightMagnitude = clamp(-1 * sin(radians(p)), 0.0, 1.0)
    nightTimeLightMagnitude = clamp(sin(radians(p)), 0.0, 1.0)

    sunlight.setColor(lightColor * dayTimeLightMagnitude)
    moonlight.setColor(lightColor * nightTimeLightMagnitude)

    if dayTimeLightMagnitude > 0.0:
        sunlight.setShadowCaster(True, SHADOW_SIZE, SHADOW_SIZE)
        render.setLight(sunlightNP)
    else:
        sunlight.setShadowCaster(False, 0, 0)
        render.setLightOff(sunlightNP)

    if nightTimeLightMagnitude > 0.0:
        moonlight.setShadowCaster(True, SHADOW_SIZE, SHADOW_SIZE)
        render.setLight(moonlightNP)
    else:
        moonlight.setShadowCaster(False, 0, 0)
        render.setLightOff(moonlightNP)

    def updateWindowLight(name):
        windowLightNP = render.find("**/" + name)
        windowLight = windowLightNP.node()
        windowLightMagnitude = nightTimeLightMagnitude**0.4
        windowLight.setColor(windowLightColor * windowLightMagnitude)
        if windowLightMagnitude <= 0.0:
            windowLight.setShadowCaster(False, 0, 0)
            render.setLightOff(windowLightNP)
        else:
            windowLight.setShadowCaster(True, SHADOW_SIZE, SHADOW_SIZE)
            render.setLight(windowLightNP)

    updateWindowLight("windowLight")
    updateWindowLight("windowLight1")
    updateWindowLight("windowLight2")

    if mixFactor >= 0.3 and mixFactor <= 0.35 and closedShutters or midnightDown:
        closedShutters = False
        shuttersAnimationCollection.play("open-shutters")
    elif mixFactor >= 0.6 and mixFactor <= 0.7 and not closedShutters or middayDown:
        closedShutters = True
        shuttersAnimationCollection.play("close-shutters")

    return p


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
        int(rgbaBits[0]), int(rgbaBits[1]), int(rgbaBits[2]), int(rgbaBits[3])
    )
    fbp.setAuxRgba(aux_rgba)
    fbp.setFloatColor(setFloatColor)
    fbp.setSrgbColor(setSrgbColor)
    fbp.setRgbColor(setRgbColor)
    fbp.setFloatDepth(True)

    buffer = graphicsEngine.makeOutput(
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
        camera = cameraNP.node()
        dr = camera.getDisplayRegion(0)
        if dr is not None:
            dr.setActive(False)
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


def generateSsaoSamples(numberOfSamples):
    def lerp(a, b, f):
        return a + f * (b - a)

    ssaoSamples = PTA_LVecBase3f()

    for i in range(numberOfSamples):
        sample = LVector3f(rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand()).normalized()

        randx = rand()
        sample[0] *= randx
        sample[1] *= randx
        sample[2] *= randx

        scale = i / numberOfSamples
        scale = lerp(0.1, 1.0, scale * scale)
        sample[0] *= scale
        sample[1] *= scale
        sample[2] *= scale
        ssaoSamples.push_back(sample)

    return ssaoSamples


def generateSsaoNoise(numberOfNoise):
    ssaoNoise = PTA_LVecBase3f()

    for i in range(numberOfNoise):
        noise = LVector3f(rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, 0.0)
        ssaoNoise.push_back(noise)

    return ssaoNoise


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


def isButtonDown(mouseWatcher, character):
    return mouseWatcher.isButtonDown(ButtonRegistry.ptr().findButton(character))


def setSoundState(sound: AudioSound, on: bool):
    if not on and sound.status() == AudioSound.PLAYING:
        sound.stop()
    elif on and sound.status() != AudioSound.PLAYING:
        sound.play()


def setSoundOff(sound: AudioSound):
    setSoundState(sound, False)


def setSoundOn(sound: AudioSound):
    setSoundState(sound, True)


base = ShowBase()
render = base.render
render2d = base.render2d
loader = base.loader
taskManager = base.taskMgr
audioManager = AudioManager.createAudioManager()
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

font = FontPool.loadFont("fonts/font.ttf")
sounds = [
    audioManager.getSound("sounds/wheel.ogg", True),
    audioManager.getSound("sounds/water.ogg", True),
]

window.setClearColorActive(True)
window.setClearDepthActive(True)
window.setClearStencilActive(True)
window.setClearColor(backgroundColor[1])
window.setClearDepth(1.0)
window.setClearStencil(0)

status = TextNode("status")
status.setFont(font)
status.setText(statusText)
status.setTextColor(statusColor)
status.setShadow(0.0, 0.06)
status.setShadowColor(statusShadowColor)
statusNP = render2d.attachNewNode(status)
statusNP.setScale(0.05)
statusNP.setPos(-0.96, 0, -0.95)

base.disableMouse()
mouseWatcher = base.mouseWatcher.node()

mainCamera = base.cam.node()
mainLens = mainCamera.getLens()
mainLens.setFov(cameraFov)
mainLens.setNearFar(cameraNear, cameraFar)

cameraNP = base.camera
cameraNP.setPos(
    calculateCameraPosition(
        cameraRotateRadius, cameraRotatePhi, cameraRotateTheta, cameraLookAt
    )
)
cameraNP.lookAt(cameraLookAt)

blankTexture = loader.loadTexture("images/blank.png")
foamPatternTexture = loader.loadTexture("images/foam-pattern.png")
stillFlowTexture = loader.loadTexture("images/still-flow.png")
upFlowTexture = loader.loadTexture("images/up-flow.png")
colorLookupTableTextureN = loader.loadTexture("images/lookup-table-neutral.png")
colorLookupTableTexture0 = loader.loadTexture("images/lookup-table-0.png")
colorLookupTableTexture1 = loader.loadTexture("images/lookup-table-1.png")
smokeTexture = loader.loadTexture("images/smoke.png")
colorNoiseTexture = loader.loadTexture("images/color-noise.png")

setTextureToNearestAndClamp(colorLookupTableTextureN);
setTextureToNearestAndClamp(colorLookupTableTexture0);
setTextureToNearestAndClamp(colorLookupTableTexture1);

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
mainCamera.setInitialState(mainCameraNP.getState())

isWaterNP = NodePath("isWater")
isWaterNP.setShaderInput("isWater", (1.0, 1.0))
isWaterNP.setShaderInput("flowTexture", upFlowTexture)
isWaterNP.setShaderInput("foamPatternTexture", foamPatternTexture)

isSmokeNP = NodePath("isSmoke")
isSmokeNP.setShaderInput("isSmoke", (1.0, 1.0))
isSmokeNP.setShaderInput("isParticle", (1.0, 1.0))

previousViewWorldMat = LMatrix4()
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

framebufferTextureArguments.aux_rgba = 4
framebufferTextureArguments.name = "geometry1"

geometryFrameBufferTexture1 = generateFramebufferTexture(framebufferTextureArguments)
geometryBuffer1 = geometryFrameBufferTexture1.buffer
geometryCamera1 = geometryFrameBufferTexture1.camera
geometryNP1: NodePath = geometryFrameBufferTexture1.shaderNP
geometryBuffer1.addRenderTexture(
    Texture(), GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_aux_rgba_0
)
geometryBuffer1.setClearActive(3, True)
geometryBuffer1.setClearValue(3, framebufferTextureArguments.clearColor)
geometryBuffer1.addRenderTexture(
    Texture(), GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_aux_rgba_1
)
geometryBuffer1.setClearActive(4, True)
geometryBuffer1.setClearValue(4, framebufferTextureArguments.clearColor)
geometryBuffer1.addRenderTexture(
    Texture(), GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_aux_rgba_2
)
geometryBuffer1.setClearActive(5, True)
geometryBuffer1.setClearValue(5, framebufferTextureArguments.clearColor)
geometryBuffer1.addRenderTexture(
    Texture(), GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_aux_rgba_3
)
geometryBuffer1.setClearActive(6, True)
geometryBuffer1.setClearValue(6, framebufferTextureArguments.clearColor)
geometryNP1.setShader(geometryBufferShader1)
geometryNP1.setShaderInput("normalMapsEnabled", normalMapsEnabled)
geometryNP1.setShaderInput("flowTexture", stillFlowTexture)
geometryNP1.setShaderInput("foamPatternTexture", blankTexture)
geometryNP1.setShaderInput("flowMapsEnabled", flowMapsEnabled)
geometryCamera1.setInitialState(geometryNP1.getState())
geometryCamera1.setTagStateKey("geometryBuffer1")
geometryCamera1.setTagState("isWater", isWaterNP.getState())
geometryCamera1.setCameraMask(BitMask32.bit(2))
positionTexture1 = geometryBuffer1.getTexture(0)
normalTexture1 = geometryBuffer1.getTexture(1)
reflectionMaskTexture = geometryBuffer1.getTexture(2)
refractionMaskTexture = geometryBuffer1.getTexture(3)
foamMaskTexture = geometryBuffer1.getTexture(4)
geometryCameraLens1 = geometryCamera1.getLens()
waterNP.setTag("geometryBuffer1", "isWater")
smokeNP.hide(BitMask32.bit(2))

framebufferTextureArguments.aux_rgba = 1
framebufferTextureArguments.name = "geometry2"

geometryFrameBufferTexture2 = generateFramebufferTexture(framebufferTextureArguments)
geometryBuffer2 = geometryFrameBufferTexture2.buffer
geometryCamera2 = geometryFrameBufferTexture2.camera
geometryNP2: NodePath = geometryFrameBufferTexture2.shaderNP
geometryBuffer2.addRenderTexture(
    Texture(), GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_aux_rgba_0
)
geometryBuffer2.setClearActive(3, True)
geometryBuffer2.setClearValue(3, framebufferTextureArguments.clearColor)
geometryBuffer2.setSort(geometryBuffer1.getSort() + 1)
geometryNP2.setShader(geometryBufferShader2)
geometryNP2.setShaderInput("isSmoke", (0, 0))
geometryNP2.setShaderInput("positionTexture", positionTexture1)
geometryCamera2.setInitialState(geometryNP2.getState())
geometryCamera2.setTagStateKey("geometryBuffer2")
geometryCamera2.setTagState("isSmoke", isSmokeNP.getState())
smokeNP.setTag("geometryBuffer2", "isSmoke")
positionTexture2 = geometryBuffer2.getTexture(0)
smokeMaskTexture = geometryBuffer2.getTexture(1)
geometryCameraLens2 = geometryCamera2.getLens()

framebufferTextureArguments.rgbaBits = rgba8
framebufferTextureArguments.aux_rgba = 0
framebufferTextureArguments.clearColor = LColor(0, 0, 0, 0)
framebufferTextureArguments.setFloatColor = False
framebufferTextureArguments.useScene = False
framebufferTextureArguments.name = "fog"

fogFrameBufferTexture = generateFramebufferTexture(framebufferTextureArguments)
fogBuffer = fogFrameBufferTexture.buffer
fogCamera = fogFrameBufferTexture.camera
fogNP: NodePath = fogFrameBufferTexture.shaderNP
fogBuffer.setSort(geometryBuffer2.getSort() + 1)
fogNP.setShader(fogShader)
fogNP.setShaderInput("pi", PI_SHADER_INPUT)
fogNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
fogNP.setShaderInput("backgroundColor0", backgroundColor[0])
fogNP.setShaderInput("backgroundColor1", backgroundColor[1])
fogNP.setShaderInput("positionTexture0", positionTexture1)
fogNP.setShaderInput("positionTexture1", positionTexture2)
fogNP.setShaderInput("smokeMaskTexture", smokeMaskTexture)
fogNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
fogNP.setShaderInput(
    "origin", cameraNP.getRelativePoint(render, environmentNP.getPos())
)
fogNP.setShaderInput("nearFar", LVecBase2f(fogNear, fogFar))
fogNP.setShaderInput("enabled", fogEnabled)
fogCamera.setInitialState(fogNP.getState())
fogTexture = fogBuffer.getTexture(0)

framebufferTextureArguments.clearColor = LColor(1, 1, 1, 0)
framebufferTextureArguments.name = "ssao"

ssaoFrameBufferTexture = generateFramebufferTexture(framebufferTextureArguments)
ssaoBuffer = ssaoFrameBufferTexture.buffer
ssaoCamera = ssaoFrameBufferTexture.camera
ssaoNP: NodePath = ssaoFrameBufferTexture.shaderNP
ssaoBuffer.setSort(geometryBuffer0.getSort() + 1)
ssaoNP.setShader(ssaoShader)
ssaoNP.setShaderInput("positionTexture", positionTexture0)
ssaoNP.setShaderInput("normalTexture", normalTexture0)
ssaoNP.setShaderInput("samples", generateSsaoSamples(SSAO_SAMPLES))
ssaoNP.setShaderInput("noise", generateSsaoNoise(SSAO_NOISE))
ssaoNP.setShaderInput("lensProjection", geometryCameraLens0.getProjectionMat())
ssaoNP.setShaderInput("enabled", ssaoEnabled)
ssaoCamera.setInitialState(ssaoNP.getState())

framebufferTextureArguments.name = "ssaoBlur"

ssaoBlurFrameBufferTexture = generateFramebufferTexture(framebufferTextureArguments)
ssaoBlurBuffer = ssaoBlurFrameBufferTexture.buffer
ssaoBlurCamera = ssaoBlurFrameBufferTexture.camera
ssaoBlurNP: NodePath = ssaoBlurFrameBufferTexture.shaderNP
ssaoBlurBuffer.setSort(ssaoBuffer.getSort() + 1)
ssaoBlurNP.setShader(kuwaharaFilterShader)
ssaoBlurNP.setShaderInput("colorTexture", ssaoBuffer.getTexture(0))
ssaoBlurNP.setShaderInput("parameters", LVecBase2f(1, 0))
ssaoBlurCamera.setInitialState(ssaoBlurNP.getState())
ssaoBlurTexture = ssaoBlurBuffer.getTexture(0)

framebufferTextureArguments.rgbaBits = rgba16
framebufferTextureArguments.clearColor = LColor(0, 0, 0, 0)
framebufferTextureArguments.name = "refractionUv"

refractionUvFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
refractionUvBuffer = refractionUvFramebufferTexture.buffer
refractionUvCamera = refractionUvFramebufferTexture.camera
refractionUvNP: NodePath = refractionUvFramebufferTexture.shaderNP
refractionUvBuffer.setSort(geometryBuffer1.getSort() + 1)
refractionUvNP.setShader(screenSpaceRefractionShader)
refractionUvNP.setShaderInput("positionFromTexture", positionTexture1)
refractionUvNP.setShaderInput("positionToTexture", positionTexture0)
refractionUvNP.setShaderInput("normalFromTexture", normalTexture1)
refractionUvNP.setShaderInput("lensProjection", geometryCameraLens0.getProjectionMat())
refractionUvNP.setShaderInput("enabled", refractionEnabled)
refractionUvNP.setShaderInput("rior", rior)
refractionUvCamera.setInitialState(refractionUvNP.getState())
refractionUvTexture = refractionUvBuffer.getTexture(0)

framebufferTextureArguments.name = "reflectionUv"

reflectionUvFrameBufferTexture = generateFramebufferTexture(framebufferTextureArguments)
reflectionUvBuffer = reflectionUvFrameBufferTexture.buffer
reflectionUvCamera = reflectionUvFrameBufferTexture.camera
reflectionUvNP: NodePath = reflectionUvFrameBufferTexture.shaderNP
reflectionUvBuffer.setSort(geometryBuffer1.getSort() + 1)
reflectionUvNP.setShader(screenSpaceReflectionShader)
reflectionUvNP.setShaderInput("positionTexture", positionTexture1)
reflectionUvNP.setShaderInput("normalTexture", normalTexture1)
reflectionUvNP.setShaderInput("maskTexture", reflectionMaskTexture)
reflectionUvNP.setShaderInput("lensProjection", geometryCameraLens0.getProjectionMat())
reflectionUvNP.setShaderInput("enabled", reflectionEnabled)
reflectionUvCamera.setInitialState(reflectionUvNP.getState())
reflectionUvTexture = reflectionUvBuffer.getTexture(0)

framebufferTextureArguments.rgbaBits = rgba8
framebufferTextureArguments.aux_rgba = 1
framebufferTextureArguments.useScene = True
framebufferTextureArguments.name = "base"

baseFrameBufferTexture = generateFramebufferTexture(framebufferTextureArguments)
baseBuffer = baseFrameBufferTexture.buffer
baseCamera = baseFrameBufferTexture.camera
baseNP: NodePath = baseFrameBufferTexture.shaderNP
baseBuffer.addRenderTexture(
    Texture(), GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_aux_rgba_0
)
baseBuffer.setClearActive(3, True)
baseBuffer.setClearValue(3, framebufferTextureArguments.clearColor)
baseBuffer.setSort(max(ssaoBlurBuffer.getSort() + 1, UNSORTED_RENDER_SORT_ORDER + 1))
baseNP.setShader(baseShader)
baseNP.setShaderInput("pi", PI_SHADER_INPUT)
baseNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
baseNP.setShaderInput("ssaoBlurTexture", ssaoBlurTexture)
baseNP.setShaderInput("flowTexture", stillFlowTexture)
baseNP.setShaderInput("normalMapsEnabled", normalMapsEnabled)
baseNP.setShaderInput("blinnPhongEnabled", blinnPhongEnabled)
baseNP.setShaderInput("fresnelEnabled", fresnelEnabled)
baseNP.setShaderInput("rimLightEnabled", rimLightEnabled)
baseNP.setShaderInput("celShadingEnabled", celShadingEnabled)
baseNP.setShaderInput("flowMapsEnabled", flowMapsEnabled)
baseNP.setShaderInput("specularColor", LVecBase2f(0, 0))
baseNP.setShaderInput("isParticle", LVecBase2f(0, 0))
baseNP.setShaderInput("isWater", LVecBase2f(0, 0))
baseNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
baseCamera.setInitialState(baseNP.getState())
baseCamera.setTagStateKey("baseBuffer")
baseCamera.setTagState("isParticle", isSmokeNP.getState())
baseCamera.setTagState("isWater", isWaterNP.getState())
baseCamera.setCameraMask(BitMask32.bit(6))
smokeNP.setTag("baseBuffer", "isParticle")
waterNP.setTag("baseBuffer", "isWater")
baseTexture = baseBuffer.getTexture(0)
specularTexture = baseBuffer.getTexture(1)

framebufferTextureArguments.aux_rgba = 0
framebufferTextureArguments.useScene = False
framebufferTextureArguments.name = "refraction"
refractionFrameBufferTexture = generateFramebufferTexture(framebufferTextureArguments)
refractionBuffer = refractionFrameBufferTexture.buffer
refractionCamera = refractionFrameBufferTexture.camera
refractionNP: NodePath = refractionFrameBufferTexture.shaderNP
refractionBuffer.setSort(baseBuffer.getSort() + 1)
refractionNP.setShader(refractionShader)
refractionNP.setShaderInput("pi", PI_SHADER_INPUT)
refractionNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
refractionNP.setShaderInput("uvTexture", refractionUvTexture)
refractionNP.setShaderInput("maskTexture", refractionMaskTexture)
refractionNP.setShaderInput("positionFromTexture", positionTexture1)
refractionNP.setShaderInput("positionToTexture", positionTexture0)
refractionNP.setShaderInput("backgroundColorTexture", baseTexture)
refractionNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
refractionCamera.setInitialState(refractionNP.getState())
refractionTexture = refractionBuffer.getTexture(0)

framebufferTextureArguments.name = "foam"
foamFrameBufferTexture = generateFramebufferTexture(framebufferTextureArguments)
foamBuffer = foamFrameBufferTexture.buffer
foamCamera = foamFrameBufferTexture.camera
foamNP: NodePath = foamFrameBufferTexture.shaderNP
foamBuffer.setSort(geometryBuffer1.getSort() + 1)
foamNP.setShader(foamShader)
foamNP.setShaderInput("pi", PI_SHADER_INPUT)
foamNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
foamNP.setShaderInput("maskTexture", foamMaskTexture)
foamNP.setShaderInput("foamDepth", foamDepth)
foamNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
foamNP.setShaderInput("viewWorldMat", currentViewWorldMat)
foamNP.setShaderInput("positionFromTexture", positionTexture1)
foamNP.setShaderInput("positionToTexture", positionTexture0)
foamCamera.setInitialState(foamNP.getState())
foamTexture = foamBuffer.getTexture(0)

framebufferTextureArguments.name = "reflectionColor"

reflectionColorFramebufferTexture = generateFramebufferTexture(
    framebufferTextureArguments
)
reflectionColorBuffer = reflectionColorFramebufferTexture.buffer
reflectionColorCamera = reflectionColorFramebufferTexture.camera
reflectionColorNP = reflectionColorFramebufferTexture.shaderNP
reflectionColorBuffer.setSort(refractionBuffer.getSort() + 1)
reflectionColorNP.setShader(reflectionColorShader)
reflectionColorNP.setShaderInput("colorTexture", refractionTexture)
reflectionColorNP.setShaderInput("uvTexture", reflectionUvTexture)
reflectionColorCamera.setInitialState(reflectionColorNP.getState())
reflectionColorTexture = reflectionColorBuffer.getTexture(0)

framebufferTextureArguments.name = "reflectionColorBlur"

reflectionColorBlurFramebufferTexture = generateFramebufferTexture(
    framebufferTextureArguments
)
reflectionColorBlurBuffer = reflectionColorBlurFramebufferTexture.buffer
reflectionColorBlurCamera = reflectionColorBlurFramebufferTexture.camera
reflectionColorBlurNP = reflectionColorBlurFramebufferTexture.shaderNP
reflectionColorBlurBuffer.setSort(reflectionColorBuffer.getSort() + 1)
reflectionColorBlurNP.setShader(boxBlurShader)
reflectionColorBlurNP.setShaderInput("colorTexture", reflectionColorTexture)
reflectionColorBlurNP.setShaderInput("parameters", LVecBase2f(8, 1))
reflectionColorBlurCamera.setInitialState(reflectionColorBlurNP.getState())
reflectionColorBlurTexture = reflectionColorBlurBuffer.getTexture(0)

framebufferTextureArguments.name = "reflection"

reflectionFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
reflectionBuffer = reflectionFramebufferTexture.buffer
reflectionNP = reflectionFramebufferTexture.shaderNP
reflectionBuffer.setSort(reflectionColorBlurBuffer.getSort() + 1)
reflectionNP.setShader(reflectionShader)
reflectionNP.setShaderInput("colorTexture", reflectionColorTexture)
reflectionNP.setShaderInput("colorBlurTexture", reflectionColorBlurTexture)
reflectionNP.setShaderInput("maskTexture", reflectionMaskTexture)
reflectionFramebufferTexture.camera.setInitialState(reflectionNP.getState())
reflectionTexture = reflectionBuffer.getTexture()

framebufferTextureArguments.name = "baseCombine"

baseCombineFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
baseCombineBuffer = baseCombineFramebufferTexture.buffer
baseCombineCamera = baseCombineFramebufferTexture.camera
baseCombineNP = baseCombineFramebufferTexture.shaderNP
baseCombineBuffer.setSort(reflectionBuffer.getSort() + 1)
baseCombineNP.setShader(baseCombineShader)
baseCombineNP.setShaderInput("baseTexture", baseTexture)
baseCombineNP.setShaderInput("refractionTexture", refractionTexture)
baseCombineNP.setShaderInput("foamTexture", foamTexture)
baseCombineNP.setShaderInput("reflectionTexture", reflectionTexture)
baseCombineNP.setShaderInput("specularTexture", specularTexture)
baseCombineCamera.setInitialState(baseCombineNP.getState())
baseCombineTexture = baseCombineBuffer.getTexture()

framebufferTextureArguments.name = "sharpen"

sharpenFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
sharpenBuffer = sharpenFramebufferTexture.buffer
sharpenNP = sharpenFramebufferTexture.shaderNP
sharpenBuffer.setSort(baseCombineBuffer.getSort() + 1)
sharpenNP.setShader(sharpenShader)
sharpenNP.setShaderInput("colorTexture", baseCombineTexture)
sharpenNP.setShaderInput("enabled", sharpenEnabled)
sharpenCamera = sharpenFramebufferTexture.camera
sharpenCamera.setInitialState(sharpenNP.getState())
sharpenTexture = sharpenBuffer.getTexture()

framebufferTextureArguments.name = "posterize"

posterizeFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
posterizeBuffer = posterizeFramebufferTexture.buffer
posterizeNP = posterizeFramebufferTexture.shaderNP
posterizeBuffer.setSort(sharpenBuffer.getSort() + 1)
posterizeNP.setShader(posterizeShader)
posterizeNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
posterizeNP.setShaderInput("colorTexture", sharpenTexture)
posterizeNP.setShaderInput("positionTexture", positionTexture2)
posterizeNP.setShaderInput("enabled", posterizeEnabled)
posterizeCamera = posterizeFramebufferTexture.camera
posterizeCamera.setInitialState(posterizeNP.getState())
posterizeTexture = posterizeBuffer.getTexture()

framebufferTextureArguments.name = "bloom"

bloomFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
bloomBuffer = bloomFramebufferTexture.buffer
bloomCamera = bloomFramebufferTexture.camera
bloomNP = bloomFramebufferTexture.shaderNP
bloomBuffer.setSort(posterizeBuffer.getSort() + 1)
bloomNP.setShader(bloomShader)
bloomNP.setShaderInput("colorTexture", posterizeTexture)
bloomNP.setShaderInput("enabled", bloomEnabled)
bloomCamera.setInitialState(bloomNP.getState())
bloomTexture = bloomBuffer.getTexture()

framebufferTextureArguments.name = "sceneCombine"

sceneCombineFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
sceneCombineBuffer = sceneCombineFramebufferTexture.buffer
sceneCombineCamera = sceneCombineFramebufferTexture.camera
sceneCombineNP = sceneCombineFramebufferTexture.shaderNP
sceneCombineBuffer.setSort(bloomBuffer.getSort() + 1)
sceneCombineNP.setShader(sceneCombineShader)
sceneCombineNP.setShaderInput("pi", PI_SHADER_INPUT)
sceneCombineNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
sceneCombineNP.setShaderInput("lookupTableTextureN", colorLookupTableTextureN)
sceneCombineNP.setShaderInput("backgroundColor0", backgroundColor[0])
sceneCombineNP.setShaderInput("backgroundColor1", backgroundColor[1])
sceneCombineNP.setShaderInput("baseTexture", posterizeTexture)
sceneCombineNP.setShaderInput("bloomTexture", bloomTexture)
sceneCombineNP.setShaderInput("fogTexture", fogTexture)
sceneCombineNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
sceneCombineTexture = sceneCombineBuffer.getTexture()
sceneCombineCamera.setInitialState(sceneCombineNP.getState())

framebufferTextureArguments.clearColor = backgroundColor[1]
framebufferTextureArguments.name = "outOfFocus"

outOfFocusFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
outOfFocusBuffer = outOfFocusFramebufferTexture.buffer
outOfFocusCamera = outOfFocusFramebufferTexture.camera
outOfFocusNP = outOfFocusFramebufferTexture.shaderNP
outOfFocusBuffer.setSort(sceneCombineBuffer.getSort() + 1)
outOfFocusNP.setShader(boxBlurShader)
outOfFocusNP.setShaderInput("colorTexture", sceneCombineTexture)
outOfFocusNP.setShaderInput("parameters", LVecBase2f(2, 2))
outOfFocusCamera.setInitialState(outOfFocusNP.getState())
outOfFocusTexture = outOfFocusBuffer.getTexture()

framebufferTextureArguments.name = "dilatedOutOfFocus"

dilatedOutOfFocusFramebufferTexture = generateFramebufferTexture(
    framebufferTextureArguments
)
dilatedOutOfFocusBuffer = dilatedOutOfFocusFramebufferTexture.buffer
dilatedOutOfFocusCamera = dilatedOutOfFocusFramebufferTexture.camera
dilatedOutOfFocusNP = dilatedOutOfFocusFramebufferTexture.shaderNP
dilatedOutOfFocusBuffer.setSort(outOfFocusBuffer.getSort() + 1)
dilatedOutOfFocusNP.setShader(dilationShader)
dilatedOutOfFocusNP.setShaderInput("colorTexture", outOfFocusTexture)
dilatedOutOfFocusNP.setShaderInput("parameters", LVecBase2f(4, 2))
dilatedOutOfFocusCamera.setInitialState(dilatedOutOfFocusNP.getState())
dilatedOutOfFocusTexture = dilatedOutOfFocusBuffer.getTexture()

framebufferTextureArguments.aux_rgba = 1
framebufferTextureArguments.name = "depthOfField"

depthOfFieldFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
depthOfFieldBuffer = depthOfFieldFramebufferTexture.buffer
depthOfFieldNP = depthOfFieldFramebufferTexture.shaderNP
depthOfFieldBuffer.addRenderTexture(
    Texture(), GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_aux_rgba_0
)
depthOfFieldBuffer.setClearActive(3, True)
depthOfFieldBuffer.setClearValue(3, framebufferTextureArguments.clearColor)
depthOfFieldBuffer.setSort(dilatedOutOfFocusBuffer.getSort() + 1)
depthOfFieldNP.setShader(depthOfFieldShader)
depthOfFieldNP.setShaderInput("positionTexture", positionTexture0)
depthOfFieldNP.setShaderInput("focusTexture", sceneCombineTexture)
depthOfFieldNP.setShaderInput("outOfFocusTexture", dilatedOutOfFocusTexture)
depthOfFieldNP.setShaderInput("mouseFocusPoint", mouseFocusPoint)
depthOfFieldNP.setShaderInput("nearFar", cameraNearFar)
depthOfFieldNP.setShaderInput("enabled", depthOfFieldEnabled)
depthOfFieldCamera = depthOfFieldFramebufferTexture.camera
depthOfFieldCamera.setInitialState(depthOfFieldNP.getState())
depthOfFieldTexture0 = depthOfFieldBuffer.getTexture(0)
depthOfFieldTexture1 = depthOfFieldBuffer.getTexture(1)

framebufferTextureArguments.aux_rgba = 0
framebufferTextureArguments.name = "outline"

outlineFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
outlineBuffer = outlineFramebufferTexture.buffer
outlineCamera = outlineFramebufferTexture.camera
outlineNP = outlineFramebufferTexture.shaderNP
outlineBuffer.setSort(depthOfFieldBuffer.getSort() + 1)
outlineNP.setShader(outlineShader)
outlineNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
outlineNP.setShaderInput("positionTexture", positionTexture0)
outlineNP.setShaderInput("colorTexture", depthOfFieldTexture0)
outlineNP.setShaderInput("noiseTexture", colorNoiseTexture)
outlineNP.setShaderInput("depthOfFieldTexture", depthOfFieldTexture1)
outlineNP.setShaderInput("fogTexture", fogTexture)
outlineNP.setShaderInput("nearFar", cameraNearFar)
outlineNP.setShaderInput("enabled", outlineEnabled)
outlineCamera.setInitialState(outlineNP.getState())
outlineTexture = outlineBuffer.getTexture()

framebufferTextureArguments.name = "painterly"

painterlyFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
painterlyBuffer = painterlyFramebufferTexture.buffer
painterlyNP = painterlyFramebufferTexture.shaderNP
painterlyBuffer.setSort(outlineBuffer.getSort() + 1)
painterlyNP.setShader(kuwaharaFilterShader)
painterlyNP.setShaderInput("colorTexture", outlineTexture)
painterlyNP.setShaderInput("parameters", LVecBase2f(0, 0))
painterlyCamera = painterlyFramebufferTexture.camera
painterlyCamera.setInitialState(painterlyNP.getState())
painterlyTexture = painterlyBuffer.getTexture()

framebufferTextureArguments.name = "pixelize"

pixelizeFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
pixelizeBuffer = pixelizeFramebufferTexture.buffer
pixelizeNP = pixelizeFramebufferTexture.shaderNP
pixelizeBuffer.setSort(painterlyBuffer.getSort() + 1)
pixelizeNP.setShader(pixelizeShader)
pixelizeNP.setShaderInput("colorTexture", painterlyTexture)
pixelizeNP.setShaderInput("positionTexture", positionTexture2)
pixelizeNP.setShaderInput("parameters", LVecBase2f(5, 0))
pixelizeNP.setShaderInput("enabled", pixelizeEnabled)
pixelizeCamera = pixelizeFramebufferTexture.camera
pixelizeCamera.setInitialState(pixelizeNP.getState())
pixelizeTexture = pixelizeBuffer.getTexture()

framebufferTextureArguments.name = "motionBlur"

motionBlurFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
motionBlurBuffer = motionBlurFramebufferTexture.buffer
motionBlurNP = motionBlurFramebufferTexture.shaderNP
motionBlurBuffer.setSort(pixelizeBuffer.getSort() + 1)
motionBlurNP.setShader(motionBlurShader)
motionBlurNP.setShaderInput("previousViewWorldMat", previousViewWorldMat)
motionBlurNP.setShaderInput("worldViewMat", render.getTransform(cameraNP).getMat())
motionBlurNP.setShaderInput("lensProjection", geometryCameraLens2.getProjectionMat())
motionBlurNP.setShaderInput("positionTexture", positionTexture2)
motionBlurNP.setShaderInput("colorTexture", pixelizeTexture)
motionBlurNP.setShaderInput("motionBlurEnabled", motionBlurEnabled)
motionBlurNP.setShaderInput("parameters", LVecBase2f(2, 1.0))
motionBlurCamera = motionBlurFramebufferTexture.camera
motionBlurCamera.setInitialState(motionBlurNP.getState())
motionBlurTexture = motionBlurBuffer.getTexture()

framebufferTextureArguments.name = "filmGrain"

filmGrainFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
filmGrainBuffer = filmGrainFramebufferTexture.buffer
filmGrainNP = filmGrainFramebufferTexture.shaderNP
filmGrainBuffer.setSort(motionBlurBuffer.getSort() + 1)
filmGrainNP.setShader(filmGrainShader)
filmGrainNP.setShaderInput("pi", PI_SHADER_INPUT)
filmGrainNP.setShaderInput("colorTexture", motionBlurTexture)
filmGrainNP.setShaderInput("enabled", filmGrainEnabled)
filmGrainCamera = filmGrainFramebufferTexture.camera
filmGrainCamera.setInitialState(filmGrainNP.getState())
filmGrainTexture = filmGrainBuffer.getTexture()

framebufferTextureArguments.name = "lookupTable"

lookupTableFramebufferTexture = generateFramebufferTexture(framebufferTextureArguments)
lookupTableBuffer = lookupTableFramebufferTexture.buffer
lookupTableNP = lookupTableFramebufferTexture.shaderNP
lookupTableBuffer.setSort(filmGrainBuffer.getSort() + 1)
lookupTableNP.setShader(lookupTableShader)
lookupTableNP.setShaderInput("pi", PI_SHADER_INPUT)
lookupTableNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
lookupTableNP.setShaderInput("colorTexture", filmGrainTexture)
lookupTableNP.setShaderInput("lookupTableTextureN", colorLookupTableTextureN)
lookupTableNP.setShaderInput("lookupTableTexture0", colorLookupTableTexture0)
lookupTableNP.setShaderInput("lookupTableTexture1", colorLookupTableTexture1)
lookupTableNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
lookupTableNP.setShaderInput("enabled", lookupTableEnabled)
lookupTableCamera = lookupTableFramebufferTexture.camera
lookupTableCamera.setInitialState(lookupTableNP.getState())
lookupTableTexture = lookupTableBuffer.getTexture()

framebufferTextureArguments.name = "gammaCorrection"

gammaCorrectionFramebufferTexture = generateFramebufferTexture(
    framebufferTextureArguments
)
gammaCorrectionBuffer = gammaCorrectionFramebufferTexture.buffer
gammaCorrectionNP = gammaCorrectionFramebufferTexture.shaderNP
gammaCorrectionBuffer.setSort(lookupTableBuffer.getSort() + 1)
gammaCorrectionNP.setShader(gammaCorrectionShader)
gammaCorrectionNP.setShaderInput("gamma", GAMMA_SHADER_INPUT)
gammaCorrectionNP.setShaderInput("colorTexture", lookupTableTexture)
gammaCorrectionCamera = gammaCorrectionFramebufferTexture.camera
gammaCorrectionCamera.setInitialState(gammaCorrectionNP.getState())
gammaCorrectionTexture = gammaCorrectionBuffer.getTexture()

framebufferTextureArguments.name = "chromaticAberration"

chromaticAberrationFramebufferTexture = generateFramebufferTexture(
    framebufferTextureArguments
)
chromaticAberrationBuffer = chromaticAberrationFramebufferTexture.buffer
chromaticAberrationNP = chromaticAberrationFramebufferTexture.shaderNP
chromaticAberrationBuffer.setSort(gammaCorrectionBuffer.getSort() + 1)
chromaticAberrationNP.setShader(chromaticAberrationShader)
chromaticAberrationNP.setShaderInput("mouseFocusPoint", mouseFocusPoint)
chromaticAberrationNP.setShaderInput("colorTexture", gammaCorrectionTexture)
chromaticAberrationNP.setShaderInput("enabled", chromaticAberrationEnabled)
chromaticAberrationCamera = chromaticAberrationFramebufferTexture.camera
chromaticAberrationCamera.setInitialState(chromaticAberrationNP.getState())

graphicsOutput.setSort(chromaticAberrationBuffer.getSort() + 1)

showBufferIndex = 0

bufferArray = [
    ("Positions 0", geometryBuffer0, 0),
    ("Normals 0", geometryBuffer0, 1),
    ("Positions 1", geometryBuffer1, 0),
    ("Normals 1", geometryBuffer1, 1),
    ("Reflection Mask", geometryBuffer1, 2),
    ("Refraction Mask", geometryBuffer1, 3),
    ("Foam Mask", geometryBuffer1, 4),
    ("Positions 2", geometryBuffer2, 0),
    ("Smoke Mask", geometryBuffer2, 1),
    ("SSAO", ssaoBuffer, 0),
    ("SSAO Blur", ssaoBlurBuffer, 0),
    ("Refraction UV", refractionUvBuffer, 0),
    ("Refraction", refractionBuffer, 0),
    ("Reflection UV", reflectionUvBuffer, 0),
    ("Reflection Color", reflectionColorBuffer, 0),
    ("Reflection Blur", reflectionColorBlurBuffer, 0),
    ("Reflection", reflectionBuffer, 0),
    ("Foam", foamBuffer, 0),
    ("Base", baseBuffer, 0),
    ("Specular", baseBuffer, 1),
    ("Base Combine", baseCombineBuffer, 0),
    ("Painterly", painterlyBuffer, 0),
    ("posterize", posterizeBuffer, 0),
    ("Bloom", bloomBuffer, 0),
    ("Outline", outlineBuffer, 0),
    ("Fog", fogBuffer, 0),
    ("Scene Combine", sceneCombineBuffer, 0),
    ("Out of Focus", outOfFocusBuffer, 0),
    ("Dilation", dilatedOutOfFocusBuffer, 0),
    ("Depeth of Field Blur", depthOfFieldBuffer, 1),
    ("Depeth of Field", depthOfFieldBuffer, 0),
    ("Pixelize", pixelizeBuffer, 0),
    ("Motion Blur", motionBlurBuffer, 0),
    ("Film Grain", filmGrainBuffer, 0),
    ("Lookup Table", lookupTableBuffer, 0),
    ("Gamma Correction", gammaCorrectionBuffer, 0),
    ("Chromatic Aberration", chromaticAberrationBuffer, 0),
]
showBufferIndex = len(bufferArray) - 1
showBuffer(render2d, None, bufferArray[showBufferIndex], False)

shuttersAnimationCollection.play("close-shutters")
weatherVaneAnimationCollection.loop("weather-vane-shake", True)
bannerAnimationCollection.loop("banner-swing", True)

then = time.perf_counter()
loopStartedAt = then
now = then
keyTime = then


def updateAudoManager(sceneRootNP, cameraNP):
    f = sceneRootNP.getRelativeVector(cameraNP, LVector3f.forward())
    u = sceneRootNP.getRelativeVector(cameraNP, LVector3f.up())
    v = LVector3f(0, 0, 0)
    p = cameraNP.getPos(sceneRootNP)

    audioManager.audio_3d_set_listener_attributes(
        p[0], p[1], p[2], v[0], v[1], v[2], f[0], f[1], f[2], u[0], u[1], u[2]
    )

    audioManager.update()


def beforeFrame(task):
    windowProperties = graphicsWindow.getProperties()
    if windowProperties.getMinimized():
        time.sleep(1)

    global then
    global loopStartedAt
    global now
    global keyTime
    global cameraRotatePhi
    global cameraRotateTheta
    global cameraRotateRadius
    global mouseWheelUp
    global mouseWheelDown
    global mouseNow
    global mouseFocusPoint
    global statusAlpha
    global statusText
    global fogNear
    global fogFar
    global showBufferIndex
    global ssaoEnabled
    global refractionEnabled
    global reflectionEnabled
    global bloomEnabled
    global normalMapsEnabled
    global fogEnabled
    global outlineEnabled
    global celShadingEnabled
    global lookupTableEnabled
    global fresnelEnabled
    global rimLightEnabled
    global blinnPhongEnabled
    global sharpenEnabled
    global depthOfFieldEnabled
    global painterlyEnabled
    global motionBlurEnabled
    global posterizeEnabled
    global pixelizeEnabled
    global flowMapsEnabled
    global filmGrainEnabled
    global soundEnabled
    global animateSunlight
    global chromaticAberrationEnabled
    global cameraLookAt
    global soundStarted

    now = time.perf_counter()

    if not soundStarted and now - loopStartedAt >= startSoundAt:
        for sound in sounds:
            sound.setLoop(True)
            sound.play()
        soundStarted = True

    delta = now - then

    then = now

    movement = 100 * delta

    timeSinceKey = now - keyTime
    keyDebounced = timeSinceKey >= 0.2

    cameraUpDownAdjust = 0
    cameraLeftRightAdjust = 0

    shiftDown = isButtonDown(mouseWatcher, "shift")
    tabDown = isButtonDown(mouseWatcher, "tab")

    resetDown = isButtonDown(mouseWatcher, "r")

    fogNearDown = isButtonDown(mouseWatcher, "[")
    fogFarDown = isButtonDown(mouseWatcher, "]")

    equalDown = isButtonDown(mouseWatcher, "=")
    minusDown = isButtonDown(mouseWatcher, "-")

    deleteDown = isButtonDown(mouseWatcher, "delete")

    wDown = isButtonDown(mouseWatcher, "w")
    aDown = isButtonDown(mouseWatcher, "a")
    dDown = isButtonDown(mouseWatcher, "d")
    sDown = isButtonDown(mouseWatcher, "s")
    zDown = isButtonDown(mouseWatcher, "z")
    xDown = isButtonDown(mouseWatcher, "x")

    arrowUpDown = isButtonDown(mouseWatcher, "arrow_up")
    arrowDownDown = isButtonDown(mouseWatcher, "arrow_down")
    arrowLeftDown = isButtonDown(mouseWatcher, "arrow_left")
    arrowRightDown = isButtonDown(mouseWatcher, "arrow_right")

    middayDown = isButtonDown(mouseWatcher, "1")
    midnightDown = isButtonDown(mouseWatcher, "2")
    fresnelDown = isButtonDown(mouseWatcher, "3")
    rimLightDown = isButtonDown(mouseWatcher, "4")
    particlesDown = isButtonDown(mouseWatcher, "5")
    motionBlurDown = isButtonDown(mouseWatcher, "6")
    painterlyDown = isButtonDown(mouseWatcher, "7")
    celShadingDown = isButtonDown(mouseWatcher, "8")
    lookupTableDown = isButtonDown(mouseWatcher, "9")
    blinnPhongDown = isButtonDown(mouseWatcher, "0")
    ssaoDown = isButtonDown(mouseWatcher, "y")
    outlineDown = isButtonDown(mouseWatcher, "u")
    bloomDown = isButtonDown(mouseWatcher, "i")
    normalMapsDown = isButtonDown(mouseWatcher, "o")
    fogDown = isButtonDown(mouseWatcher, "p")
    depthOfFieldDown = isButtonDown(mouseWatcher, "h")
    posterizeDown = isButtonDown(mouseWatcher, "j")
    pixelizeDown = isButtonDown(mouseWatcher, "k")
    sharpenDown = isButtonDown(mouseWatcher, "l")
    filmGrainDown = isButtonDown(mouseWatcher, "n")
    reflectionDown = isButtonDown(mouseWatcher, "m")
    refractionDown = isButtonDown(mouseWatcher, ",")
    flowMapsDown = isButtonDown(mouseWatcher, ".")
    sunlightDown = isButtonDown(mouseWatcher, "/")
    chromaticAberrationDown = isButtonDown(mouseWatcher, "\\")

    mouseLeftDown = mouseWatcher.isButtonDown(MouseButton.one())
    mouseMiddleDown = mouseWatcher.isButtonDown(MouseButton.two())
    mouseRightDown = mouseWatcher.isButtonDown(MouseButton.three())

    if wDown:
        cameraRotatePhi -= movement * 0.5

    if sDown:
        cameraRotatePhi += movement * 0.5

    if aDown:
        cameraRotateTheta += movement * 0.5

    if dDown:
        cameraRotateTheta -= movement * 0.5

    if zDown or mouseWheelUp:
        cameraRotateRadius -= movement * 4 + 50 * mouseWheelUp
        mouseWheelUp = False

    if xDown or mouseWheelDown:
        cameraRotateRadius += movement * 4 + 50 * mouseWheelDown
        mouseWheelDown = False

    if cameraRotatePhi < 1:
        cameraRotatePhi = 1
    if cameraRotatePhi > 179:
        cameraRotatePhi = 179
    if cameraRotatePhi < 0:
        cameraRotatePhi = 360 - cameraRotateTheta
    if cameraRotateTheta > 360:
        cameraRotateTheta = cameraRotateTheta - 360
    if cameraRotateTheta < 0:
        cameraRotateTheta = 360 - cameraRotateTheta
    if cameraRotateRadius < cameraNear + 5:
        cameraRotateRadius = cameraNear + 5
    if cameraRotateRadius > cameraFar - 10:
        cameraRotateRadius = cameraFar - 10

    if arrowUpDown:
        cameraUpDownAdjust = -2 * delta
    elif arrowDownDown:
        cameraUpDownAdjust = 2 * delta

    if arrowLeftDown:
        cameraLeftRightAdjust = 2 * delta
    elif arrowRightDown:
        cameraLeftRightAdjust = -2 * delta

    if mouseWatcher.hasMouse():
        mouseNow[0] = mouseWatcher.getMouseX()
        mouseNow[1] = mouseWatcher.getMouseY()

        if mouseLeftDown:
            cameraRotateTheta += (mouseThen[0] - mouseNow[0]) * movement
            cameraRotatePhi += (mouseNow[1] - mouseThen[1]) * movement
        elif mouseRightDown:
            cameraLeftRightAdjust = (mouseThen[0] - mouseNow[0]) * movement
            cameraUpDownAdjust = (mouseThen[1] - mouseNow[1]) * movement
        elif mouseMiddleDown:
            mouseFocusPoint = LVecBase2f(
                (mouseNow[0] + 1.0) / 2.0, (mouseNow[1] + 1.0) / 2.0
            )

        if not mouseLeftDown:
            mouseThen[0] = mouseNow[0]
            mouseThen[1] = mouseNow[1]

    if shiftDown and fogNearDown:
        fogNearDown += fogAdjust
        statusAlpha = 1.0
        statusText = "Fog Near " + str(fogNear)
    elif fogNearDown:
        fogNearDown -= fogAdjust
        statusAlpha = 1.0
        statusText = "Fog Near " + str(fogNear)

    if shiftDown and fogFarDown:
        fogFarDown -= fogAdjust
        statusAlpha = 1.0
        statusText = "Fog Far " + str(fogFar)
    elif fogFarDown:
        fogFarDown += fogAdjust
        statusAlpha = 1.0
        statusText = "Fog Far " + str(fogFar)

    if shiftDown and equalDown:
        rior[0] -= riorAdjust
        statusAlpha = 1.0
        statusText = "Refractive Index " + str(rior[0])
    elif equalDown:
        rior[0] += riorAdjust
        statusAlpha = 1.0
        statusText = "Refractive Index " + str(rior[0])

    rior[1] = rior[0]

    if shiftDown and minusDown:
        foamDepth[0] -= foamDepthAdjust
        if foamDepth[0] < 0.001:
            foamDepth[0] = 0.001
        statusAlpha = 1.0
        statusText = "Foam Depth " + str(foamDepth[0])
    elif minusDown:
        foamDepth[0] += foamDepthAdjust
        statusAlpha = 1.0
        statusText = "Foam Depth " + str(foamDepth[0])
    foamDepth[1] = foamDepth[0]

    if keyDebounced:
        if tabDown:
            if shiftDown:
                showBufferIndex -= 1
                if showBufferIndex < 0:
                    showBufferIndex = len(bufferArray) - 1
            else:
                showBufferIndex += 1
                if showBufferIndex >= len(bufferArray):
                    showBufferIndex = 0
            bufferName = bufferArray[showBufferIndex][0]
            showAlpha = (
                bufferName == "Outline" or bufferName == "Foam" or bufferName == "Fog"
            )
            showBuffer(render2d, statusNP, bufferArray[showBufferIndex], showAlpha)
            keyTime = now
            statusAlpha = 1.0
            statusText = bufferName + " Buffer"

        if resetDown:
            cameraRotateRadius = cameraRotateRadiusInitial
            cameraRotatePhi = cameraRotatePhiInitial
            cameraRotateTheta = cameraRotateThetaInitial
            cameraLookAt = cameraLookAtInitial

            fogNear = fogNearInitial
            fogFar = fogFarInitial

            foamDepth[0] = foamDepthInitial[0]
            foamDepth[1] = foamDepthInitial[1]
            rior[0] = riorInitial[0]
            rior[1] = riorInitial[1]

            mouseFocusPoint = mouseFocusPointInitial

            keyTime = now

            statusAlpha = 1.0
            statusText = "Reset"

        def toggleStatus(enabled, effect):
            global statusAlpha
            global statusText
            statusAlpha = 1.0
            if enabled[0] == 1:
                statusText = effect + " On"
            else:
                statusText = effect + " Off"

        if ssaoDown:
            ssaoEnabled = toggleEnableVec(ssaoEnabled)
            keyTime = now
            toggleStatus(ssaoEnabled, "SSAO")

        if refractionDown:
            refractionEnabled = toggleEnableVec(refractionEnabled)
            keyTime = now
            toggleStatus(refractionEnabled, "Refraction")

        if reflectionDown:
            reflectionenabled = toggleEnableVec(reflectionEnabled)
            keyTime = now
            toggleStatus(reflectionenabled, "Reflection")

        if bloomDown:
            bloomEnabled = toggleEnableVec(bloomEnabled)
            keyTime = now
            toggleStatus(bloomEnabled, "Bloom")

        if normalMapsDown:
            normalMapsEnabled = toggleEnableVec(normalMapsEnabled)
            keyTime = now
            toggleStatus(normalMapsEnabled, "Normal Maps")

        if fogDown:
            fogEnabled = toggleEnableVec(fogEnabled)
            keyTime = now
            toggleStatus(fogEnabled, "Fog")

        if outlineDown:
            outlineEnabled = toggleEnableVec(outlineEnabled)
            keyTime = now
            toggleStatus(outlineEnabled, "Outline")

        if celShadingDown:
            celShadingEnabled = toggleEnableVec(celShadingEnabled)
            keyTime = now
            toggleStatus(celShadingEnabled, "Cel Shading")

        if lookupTableDown:
            lookupTableEnabled = toggleEnableVec(lookupTableEnabled)
            keyTime = now
            toggleStatus(lookupTableEnabled, "Lookup Table")

        if fresnelDown:
            fresnelEnabled = toggleEnableVec(fresnelEnabled)
            keyTime = now
            toggleStatus(fresnelEnabled, "Fresnel")

        if rimLightDown:
            rimLightEnabled = toggleEnableVec(rimLightEnabled)
            keyTime = now
            toggleStatus(rimLightEnabled, "Rim Light")

        if blinnPhongDown:
            blinnPhongEnabled = toggleEnableVec(blinnPhongEnabled)
            keyTime = now
            toggleStatus(blinnPhongEnabled, "Blinn-Phong")

        if sharpenDown:
            sharpenEnabled = toggleEnableVec(sharpenEnabled)
            keyTime = now
            toggleStatus(sharpenEnabled, "Sharpen")

        if depthOfFieldDown:
            depthOfFieldEnabled = toggleEnableVec(depthOfFieldEnabled)
            keyTime = now
            toggleStatus(depthOfFieldEnabled, "Depth of Field")

        if painterlyDown:
            painterlyEnabled = toggleEnableVec(painterlyEnabled)
            keyTime = now
            toggleStatus(painterlyEnabled, "Painterly")

        if motionBlurDown:
            motionBlurEnabled = toggleEnableVec(motionBlurEnabled)
            keyTime = now
            toggleStatus(motionBlurEnabled, "Motion Blur")

        if posterizeDown:
            posterizeEnabled = toggleEnableVec(posterizeEnabled)
            keyTime = now
            toggleStatus(posterizeEnabled, "Posterize")

        if pixelizeDown:
            pixelizeEnabled = toggleEnableVec(pixelizeEnabled)
            keyTime = now
            toggleStatus(pixelizeEnabled, "Pixelize")

        if filmGrainDown:
            filmGrainEnabled = toggleEnableVec(filmGrainEnabled)
            keyTime = now
            toggleStatus(filmGrainEnabled, "Film Grain")

        if flowMapsDown:
            flowMapsEnabled = toggleEnableVec(flowMapsEnabled)
            if flowMapsEnabled[0] == 1 and soundEnabled:
                for sound in sounds:
                    setSoundOn(sound)
            elif flowMapsEnabled[0] != 1:
                for sound in sounds:
                    setSoundOff(sound)
            keyTime = now
            toggleStatus(flowMapsEnabled, "Flow Maps")

        if deleteDown:
            if soundEnabled:
                for sound in sounds:
                    setSoundOff(sound)
                soundEnabled = False
            else:
                if flowMapsEnabled[0] == 1:
                    for sound in sounds:
                        setSoundOn(sound)
                soundEnabled = True
            keyTime = now
            toggleStatus(LVecBase2f(1 if soundEnabled else 0), "Sound")

        if sunlightDown:
            animateSunlight = False if animateSunlight else True
            keyTime = now
            toggleStatus(LVecBase2f(1 if animateSunlight else 0), "Sun Animation")

        if particlesDown:
            keyTime = now
            statusAlpha = 1.0

            if smokeNP.isHidden():
                smokeNP.show()
                statusText = "Particles On"
            else:
                smokeNP.hide()
                statusText = "Particles Off"

        if chromaticAberrationDown:
            chromaticAberrationEnabled = toggleEnableVec(chromaticAberrationEnabled)
            keyTime = now
            toggleStatus(chromaticAberrationEnabled, "Chromatic Aberration")

    if flowMapsEnabled[0]:
        wheelP = wheelNP.getP()
        wheelP += -90.0 * delta
        if wheelP > 360:
            wheelP = 0
        if wheelP < 0:
            wheelP = 360
        wheelNP.setP(wheelP)

    if animateSunlight or middayDown or midnightDown:
        global sunlightP
        sunlightP = animateLights(
            render,
            shuttersAnimationCollection,
            delta,
            -360.0 / 64.0,
            middayDown,
            midnightDown,
        )

        if middayDown:
            statusAlpha = 1.0
            statusText = "Midday"
        elif midnightDown:
            statusAlpha = 1.0
            statusText = "Midnight"

    cameraLookAt = calculateCameraLookAt(
        cameraUpDownAdjust,
        cameraLeftRightAdjust,
        cameraRotatePhi,
        cameraRotateTheta,
        cameraLookAt,
    )

    cameraNP.setPos(
        calculateCameraPosition(
            cameraRotateRadius, cameraRotatePhi, cameraRotateTheta, cameraLookAt
        )
    )

    cameraNP.lookAt(cameraLookAt)

    global currentViewWorldMat
    global previousViewWorldMat
    currentViewWorldMat = cameraNP.getTransform(render).getMat()

    geometryNP0.setShaderInput("normalMapsEnabled", normalMapsEnabled)
    geometryNP0.setShaderInput("flowMapsEnabled", flowMapsEnabled)
    geometryCamera0.setInitialState(geometryNP0.getState())

    geometryNP1.setShaderInput("normalMapsEnabled", normalMapsEnabled)
    geometryNP1.setShaderInput("flowMapsEnabled", flowMapsEnabled)
    geometryCamera1.setInitialState(geometryNP1.getState())

    fogNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
    fogNP.setShaderInput(
        "origin", cameraNP.getRelativePoint(render, environmentNP.getPos())
    )
    fogNP.setShaderInput("nearFar", LVecBase2f(fogNear, fogFar))
    fogNP.setShaderInput("enabled", fogEnabled)
    fogCamera.setInitialState(fogNP.getState())

    ssaoNP.setShaderInput("lensProjection", geometryCameraLens0.getProjectionMat())
    ssaoNP.setShaderInput("enabled", ssaoEnabled)
    ssaoCamera.setInitialState(ssaoNP.getState())

    refractionUvNP.setShaderInput(
        "lensProjection", geometryCameraLens1.getProjectionMat()
    )
    refractionUvNP.setShaderInput("enabled", refractionEnabled)
    refractionUvNP.setShaderInput("rior", rior)
    refractionUvCamera.setInitialState(refractionUvNP.getState())

    reflectionUvNP.setShaderInput(
        "lensProjection", geometryCameraLens1.getProjectionMat()
    )
    reflectionUvNP.setShaderInput("enabled", reflectionEnabled)
    reflectionUvCamera.setInitialState(reflectionUvNP.getState())

    foamNP.setShaderInput("foamDepth", foamDepth)
    foamNP.setShaderInput("viewWorldMat", currentViewWorldMat)
    foamNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
    foamCamera.setInitialState(foamNP.getState())

    bloomNP.setShaderInput("enabled", bloomEnabled)
    bloomCamera.setInitialState(bloomNP.getState())

    outlineNP.setShaderInput("enabled", outlineEnabled)
    outlineCamera.setInitialState(outlineNP.getState())

    baseNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
    baseNP.setShaderInput("normalMapsEnabled", normalMapsEnabled)
    baseNP.setShaderInput("blinnPhongEnabled", blinnPhongEnabled)
    baseNP.setShaderInput("fresnelEnabled", fresnelEnabled)
    baseNP.setShaderInput("rimLightEnabled", rimLightEnabled)
    baseNP.setShaderInput("celShadingEnabled", celShadingEnabled)
    baseNP.setShaderInput("flowMapsEnabled", flowMapsEnabled)
    baseCamera.setInitialState(baseNP.getState())

    refractionNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
    refractionCamera.setInitialState(refractionNP.getState())

    sharpenNP.setShaderInput("enabled", sharpenEnabled)
    sharpenCamera.setInitialState(sharpenNP.getState())

    sceneCombineNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
    sceneCombineCamera.setInitialState(sceneCombineNP.getState())

    depthOfFieldNP.setShaderInput("mouseFocusPoint", mouseFocusPoint)
    depthOfFieldNP.setShaderInput("enabled", depthOfFieldEnabled)
    depthOfFieldCamera.setInitialState(depthOfFieldNP.getState())

    painterlyNP.setShaderInput(
        "parameters", LVecBase2f(3 if painterlyEnabled[0] == 1 else 0, 0)
    )
    painterlyCamera.setInitialState(painterlyNP.getState())

    motionBlurNP.setShaderInput("previousViewWorldMat", previousViewWorldMat)
    motionBlurNP.setShaderInput("worldViewMat", render.getTransform(cameraNP).getMat())
    motionBlurNP.setShaderInput(
        "lensProjection", geometryCameraLens1.getProjectionMat()
    )
    motionBlurNP.setShaderInput("motionBlurEnabled", motionBlurEnabled)
    motionBlurCamera.setInitialState(motionBlurNP.getState())

    posterizeNP.setShaderInput("enabled", posterizeEnabled)
    posterizeCamera.setInitialState(posterizeNP.getState())

    pixelizeNP.setShaderInput("enabled", pixelizeEnabled)
    pixelizeCamera.setInitialState(pixelizeNP.getState())

    filmGrainNP.setShaderInput("enabled", filmGrainEnabled)
    filmGrainCamera.setInitialState(filmGrainNP.getState())

    lookupTableNP.setShaderInput("enabled", lookupTableEnabled)
    lookupTableNP.setShaderInput("sunPosition", LVecBase2f(sunlightP, 0))
    lookupTableCamera.setInitialState(lookupTableNP.getState())

    chromaticAberrationNP.setShaderInput("mouseFocusPoint", mouseFocusPoint)
    chromaticAberrationNP.setShaderInput("enabled", chromaticAberrationEnabled)
    chromaticAberrationCamera.setInitialState(chromaticAberrationNP.getState())

    previousViewWorldMat = currentViewWorldMat

    statusAlpha = statusAlpha - ((1.0 / statusFadeRate) * delta)
    statusAlpha = 0.0 if statusAlpha < 0.0 else statusAlpha
    statusColor[3] = statusAlpha
    statusShadowColor[3] = statusAlpha
    status.setTextColor(statusColor)
    status.setShadowColor(statusShadowColor)
    status.setText(statusText)

    updateAudoManager(sceneRootNP, cameraNP)

    particleSystemManager.doParticles(delta)
    physicsManager.doPhysics(delta)

    return Task.cont


taskManager.add(beforeFrame, "beforeFrame")


def setMouseWheelUp():
    global mouseWheelUp
    mouseWheelUp = True


def setMouseWheelDown():
    global mouseWheelDown
    mouseWheelDown = True


physicsManager.attachLinearIntegrator(LinearEulerIntegrator())
wheelNPRelPos = wheelNP.getPos(sceneRootNP)
sounds[0].set_3d_attributes(
    wheelNPRelPos[0], wheelNPRelPos[1], wheelNPRelPos[2], 0, 0, 0
)
waterNPRelPos = waterNP.getPos(sceneRootNP)
sounds[1].set_3d_attributes(
    waterNPRelPos[0], waterNPRelPos[1], waterNPRelPos[2], 0, 0, 0
)
sounds[0].set_3d_min_distance(60)
sounds[1].set_3d_min_distance(50)

if __name__ == "__main__":
    base.run()
