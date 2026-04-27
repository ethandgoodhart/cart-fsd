import * as THREE from "three";

// ---------- Renderer / Scene ----------
const container = document.getElementById("scene-container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
scene.fog = new THREE.Fog(0xffffff, 14, 42);

const camera = new THREE.PerspectiveCamera(
    38,
    window.innerWidth / window.innerHeight,
    0.1,
    500
);
camera.position.set(0, 7.5, 13.0);
camera.lookAt(0, 0.6, -8);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

// ---------- Lighting (soft, minimal) ----------
scene.add(new THREE.HemisphereLight(0xffffff, 0xe5e7eb, 1.0));

const key = new THREE.DirectionalLight(0xffffff, 1.25);
key.position.set(6, 14, 8);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -10;
key.shadow.camera.right = 10;
key.shadow.camera.top = 10;
key.shadow.camera.bottom = -10;
key.shadow.camera.near = 1;
key.shadow.camera.far = 40;
key.shadow.bias = -0.0004;
key.shadow.radius = 6;
scene.add(key);

const fill = new THREE.DirectionalLight(0xffffff, 0.4);
fill.position.set(-5, 6, 4);
scene.add(fill);

// ---------- Road surface ----------
const roadWidth = 10;
const roadLength = 150;

const road = new THREE.Mesh(
    new THREE.PlaneGeometry(roadWidth, roadLength, 1, 1),
    new THREE.MeshStandardMaterial({
        color: 0xeaecef,
        roughness: 0.95,
        metalness: 0.0
    })
);
road.rotation.x = -Math.PI / 2;
road.position.set(0, 0, -roadLength / 2 + 18);
road.receiveShadow = true;
scene.add(road);

// outer shoulder (slightly lighter)
const shoulder = new THREE.Mesh(
    new THREE.PlaneGeometry(200, roadLength),
    new THREE.MeshStandardMaterial({ color: 0xf6f7f9, roughness: 1, metalness: 0 })
);
shoulder.rotation.x = -Math.PI / 2;
shoulder.position.set(0, -0.01, -roadLength / 2 + 18);
shoulder.receiveShadow = true;
scene.add(shoulder);

// ---------- Lane lines ----------
// Dashes are tracked so animate() can slide them down the road,
// producing the illusion of the cart moving forward. Wrapping keeps
// the visible stretch full no matter how long the simulation runs.
const DASH_LEN = 2.2;
const DASH_GAP = 2.6;
const DASH_STRIDE = DASH_LEN + DASH_GAP;
const DASH_Z_FAR = -roadLength + 18;              // spawn boundary (behind cart)
const DASH_Z_NEAR = DASH_Z_FAR + roadLength;      // wrap boundary (past camera)
const laneDashes = [];

function addLine({ x, dashed = false, color = 0x1a1b1f, width = 0.18 }) {
    const length = roadLength;
    if (!dashed) {
        const m = new THREE.Mesh(
            new THREE.PlaneGeometry(width, length),
            new THREE.MeshBasicMaterial({ color })
        );
        m.rotation.x = -Math.PI / 2;
        m.position.set(x, 0.012, DASH_Z_FAR + length / 2);
        scene.add(m);
    } else {
        const count = Math.floor(length / DASH_STRIDE);
        const geo = new THREE.PlaneGeometry(width, DASH_LEN);
        const mat = new THREE.MeshBasicMaterial({ color });
        for (let i = 0; i < count; i++) {
            const m = new THREE.Mesh(geo, mat);
            m.rotation.x = -Math.PI / 2;
            m.position.set(x, 0.012, DASH_Z_FAR + i * DASH_STRIDE + DASH_LEN / 2);
            scene.add(m);
            laneDashes.push(m);
        }
    }
}

// Edges (solid) + inner dashed lane dividers
addLine({ x: -roadWidth / 2, dashed: false, color: 0x1f2328, width: 0.2 });
addLine({ x: roadWidth / 2, dashed: false, color: 0x1f2328, width: 0.2 });
addLine({ x: -roadWidth / 6, dashed: true, color: 0x9aa0a8, width: 0.16 });
addLine({ x: roadWidth / 6, dashed: true, color: 0x9aa0a8, width: 0.16 });

// ---------- Predicted path strips ----------
// Two glowing strips flanking the cart's predicted path. Color tracks
// authority — blue when the human is driving (normal PS5 mode, or human
// override during --autosteer), gold while Autoware is in command.
// Recoloring is cheap (we mutate the existing materials) and the angle
// shown is always ps5_drive's commanded steer, so the path follows
// whoever owns the wheel without any sample race.
const PATH_SEGS = 60;
const PATH_TUBE_SEGS = 80;
const PATH_LENGTH = 70;
const PATH_TONES = {
    human:    { core: 0x1f6feb, halo: 0x4b8dff, haloOpacity: 0.12 },
    autoware: { core: 0xd4a017, halo: 0xf2c94c, haloOpacity: 0.18 },
};

function buildPathCurve(offsetX, steerDeg) {
    // Drive the bend from steering angle. At ±45° the far end sweeps
    // ~7 world-units laterally — enough to read as a decisive turn but
    // not so far it leaves the frame. Sign flip: positive steerDeg is
    // a right turn, which should curve the path to +x.
    const lateral = (steerDeg / 45) * 7;
    const pts = [];
    for (let i = 0; i <= PATH_SEGS; i++) {
        const t = i / PATH_SEGS;
        const z = 1.5 - t * PATH_LENGTH;
        // Quadratic bend: pinned at the cart, full offset at the horizon.
        // No per-segment jitter — tiny wiggles make TubeGeometry's Frenet
        // frames flip and render the tube inside-out in patches.
        const bend = (t * t) * lateral;
        pts.push(new THREE.Vector3(offsetX + bend, 0.09, z));
    }
    return new THREE.CatmullRomCurve3(pts);
}

const pathStrips = [];
function pathStrip(offsetX) {
    const curve = buildPathCurve(offsetX, 0);
    // Core is opaque — transparency was causing the strips to drop out
    // when a scrolling lane dash's centroid was closer to the camera
    // than the long tube's centroid. DoubleSide covers the Frenet-frame
    // flip glitch on curved tubes.
    const tone = PATH_TONES.human;
    const coreMat = new THREE.MeshBasicMaterial({
        color: tone.core,
        side: THREE.DoubleSide,
    });
    const haloMat = new THREE.MeshBasicMaterial({
        color: tone.halo, transparent: true, opacity: tone.haloOpacity,
        side: THREE.DoubleSide,
        depthWrite: false,
    });
    const core = new THREE.Mesh(
        new THREE.TubeGeometry(curve, PATH_TUBE_SEGS, 0.07, 8, false), coreMat,
    );
    const halo = new THREE.Mesh(
        new THREE.TubeGeometry(curve, PATH_TUBE_SEGS, 0.22, 8, false), haloMat,
    );
    core.renderOrder = 2;
    halo.renderOrder = 1;
    scene.add(core);
    scene.add(halo);
    pathStrips.push({ offsetX, core, halo });
}
pathStrip(-0.85);
pathStrip(0.85);

// State polls arrive at ~12 Hz but animate runs at ~60 Hz — without a
// filter the path snaps between steering samples. A low-pass toward
// the target angle turns each state update into a smooth sweep.
// TAU is the time constant: ~63% of the gap is closed in TAU seconds.
const STEER_SMOOTH_TAU = 0.11;
let lastSteerDeg = 0;
let displaySteerDeg = 0;
function updatePaths(targetSteerDeg, dt) {
    const alpha = 1 - Math.exp(-dt / STEER_SMOOTH_TAU);
    displaySteerDeg += (targetSteerDeg - displaySteerDeg) * alpha;
    if (Math.abs(displaySteerDeg - lastSteerDeg) < 0.08) return;
    lastSteerDeg = displaySteerDeg;
    for (const strip of pathStrips) {
        const curve = buildPathCurve(strip.offsetX, displaySteerDeg);
        const nextCore = new THREE.TubeGeometry(curve, PATH_TUBE_SEGS, 0.07, 8, false);
        const nextHalo = new THREE.TubeGeometry(curve, PATH_TUBE_SEGS, 0.22, 8, false);
        strip.core.geometry.dispose();
        strip.halo.geometry.dispose();
        strip.core.geometry = nextCore;
        strip.halo.geometry = nextHalo;
    }
}

// Recolor the path strips between blue (human-driven) and gold (Autoware-
// driven). pollState in index.html calls this every state update; we
// short-circuit no-op recolors so material.needsUpdate isn't flagged for
// nothing every frame.
let currentTone = "human";
window.__setPathTone = function (tone) {
    if (!PATH_TONES[tone] || tone === currentTone) return;
    currentTone = tone;
    const t = PATH_TONES[tone];
    for (const strip of pathStrips) {
        strip.core.material.color.setHex(t.core);
        strip.halo.material.color.setHex(t.halo);
        strip.halo.material.opacity = t.haloOpacity;
    }
};

// ============ GOLF CART ============
const cart = new THREE.Group();

const cartWhite = new THREE.MeshStandardMaterial({
    color: 0xfafbfc,
    roughness: 0.38,
    metalness: 0.05
});
const cartAccent = new THREE.MeshStandardMaterial({
    color: 0x1f2328,
    roughness: 0.45,
    metalness: 0.2
});
const seatMat = new THREE.MeshStandardMaterial({
    color: 0x2a2d33,
    roughness: 0.92
});
const seatCushion = new THREE.MeshStandardMaterial({
    color: 0x3a3e45,
    roughness: 0.85
});
const tireMat = new THREE.MeshStandardMaterial({
    color: 0x15171b,
    roughness: 0.9
});
const rimMat = new THREE.MeshStandardMaterial({
    color: 0xcbd0d8,
    roughness: 0.3,
    metalness: 0.75
});

function roundedBox(w, h, d, r, mat) {
    const shape = new THREE.Shape();
    shape.moveTo(-w / 2 + r, -h / 2);
    shape.lineTo(w / 2 - r, -h / 2);
    shape.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
    shape.lineTo(w / 2, h / 2 - r);
    shape.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
    shape.lineTo(-w / 2 + r, h / 2);
    shape.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
    shape.lineTo(-w / 2, -h / 2 + r);
    shape.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
    const geo = new THREE.ExtrudeGeometry(shape, {
        depth: d,
        bevelEnabled: true,
        bevelSegments: 3,
        bevelSize: 0.035,
        bevelThickness: 0.035
    });
    geo.translate(0, 0, -d / 2);
    const m = new THREE.Mesh(geo, mat);
    m.castShadow = true;
    m.receiveShadow = true;
    return m;
}

// Rear cargo deck (low, flat — golf cart style)
const deck = roundedBox(1.85, 0.12, 0.95, 0.06, cartWhite);
deck.position.set(0, 0.6, -1.35);
cart.add(deck);

// Short rear wall behind seat
const rearWall = roundedBox(1.85, 0.55, 0.08, 0.04, cartWhite);
rearWall.position.set(0, 0.82, -1.8);
cart.add(rearWall);

// Main chassis / body shell
const chassis = roundedBox(1.9, 0.32, 2.3, 0.1, cartWhite);
chassis.position.set(0, 0.52, 0);
cart.add(chassis);

// Front cowl (hood)
const cowl = roundedBox(1.85, 0.35, 0.9, 0.14, cartWhite);
cowl.position.set(0, 0.72, 1.05);
cart.add(cowl);

// Floorboard (dark)
const floor = new THREE.Mesh(
    new THREE.BoxGeometry(1.75, 0.05, 1.0),
    cartAccent
);
floor.position.set(0, 0.37, 0.25);
floor.castShadow = true;
floor.receiveShadow = true;
cart.add(floor);

// Seat cushion (bench)
const seat = new THREE.Mesh(
    new THREE.BoxGeometry(1.75, 0.18, 0.7),
    seatCushion
);
seat.position.set(0, 0.75, -0.45);
seat.castShadow = true;
cart.add(seat);

// Seat back
const seatBack = roundedBox(1.75, 0.7, 0.14, 0.06, seatMat);
seatBack.position.set(0, 1.2, -0.8);
seatBack.rotation.x = -0.08;
cart.add(seatBack);

// Seat divider (middle armrest-ish)
const divider = new THREE.Mesh(
    new THREE.BoxGeometry(0.04, 0.3, 0.65),
    cartAccent
);
divider.position.set(0, 1.0, -0.45);
cart.add(divider);

// Roof
const roof = roundedBox(2.0, 0.08, 2.1, 0.08, cartWhite);
roof.position.set(0, 2.0, -0.1);
cart.add(roof);

// Roof pillars
function pillar(x, z) {
    const p = new THREE.Mesh(
        new THREE.CylinderGeometry(0.045, 0.045, 1.3, 14),
        cartAccent
    );
    p.position.set(x, 1.3, z);
    p.castShadow = true;
    cart.add(p);
}
pillar(-0.92, 0.85);
pillar(0.92, 0.85);
pillar(-0.92, -1.1);
pillar(0.92, -1.1);

// Steering wheel
const steering = new THREE.Group();
const wheelRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.2, 0.028, 12, 32),
    cartAccent
);
steering.add(wheelRing);
const wheelHub = new THREE.Mesh(
    new THREE.CylinderGeometry(0.04, 0.04, 0.04, 16),
    cartAccent
);
wheelHub.rotation.x = Math.PI / 2;
steering.add(wheelHub);
const column = new THREE.Mesh(
    new THREE.CylinderGeometry(0.03, 0.03, 0.55, 12),
    cartAccent
);
column.rotation.x = Math.PI / 2 - 0.35;
column.position.set(0, -0.28, 0.15);
steering.add(column);
steering.position.set(-0.42, 1.05, 0.45);
steering.rotation.x = -0.3;
cart.add(steering);

// Dashboard panel
const dash = roundedBox(1.75, 0.22, 0.12, 0.03, cartAccent);
dash.position.set(0, 1.05, 0.7);
cart.add(dash);

// Wheels
function wheel(x, z) {
    const g = new THREE.Group();
    const tire = new THREE.Mesh(
        new THREE.CylinderGeometry(0.34, 0.34, 0.24, 28),
        tireMat
    );
    tire.rotation.z = Math.PI / 2;
    tire.castShadow = true;
    g.add(tire);
    const rim = new THREE.Mesh(
        new THREE.CylinderGeometry(0.19, 0.19, 0.25, 16),
        rimMat
    );
    rim.rotation.z = Math.PI / 2;
    g.add(rim);
    // rim spokes
    for (let i = 0; i < 5; i++) {
        const s = new THREE.Mesh(
            new THREE.BoxGeometry(0.35, 0.025, 0.06),
            rimMat
        );
        s.rotation.x = (i / 5) * Math.PI * 2;
        s.position.y = 0;
        g.add(s);
    }
    g.position.set(x, 0.34, z);
    return g;
}
cart.add(wheel(-0.92, 0.9));
cart.add(wheel(0.92, 0.9));
cart.add(wheel(-0.92, -1.1));
cart.add(wheel(0.92, -1.1));

// Front bumper
const bumper = new THREE.Mesh(
    new THREE.BoxGeometry(1.85, 0.08, 0.06),
    cartAccent
);
bumper.position.set(0, 0.55, 1.52);
cart.add(bumper);

// Headlights
function headlight(x) {
    const h = new THREE.Mesh(
        new THREE.CircleGeometry(0.09, 24),
        new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: 0xfff7e0,
            emissiveIntensity: 0.6,
            roughness: 0.2
        })
    );
    h.position.set(x, 0.78, 1.51);
    h.rotation.y = 0; // facing +Z locally
    cart.add(h);
    const socket = new THREE.Mesh(
        new THREE.RingGeometry(0.09, 0.11, 24),
        new THREE.MeshBasicMaterial({ color: 0x1f2328, side: THREE.DoubleSide })
    );
    socket.position.set(x, 0.78, 1.512);
    cart.add(socket);
}
headlight(-0.7);
headlight(0.7);

// Rear reflector / brake panel
function tail(x) {
    const r = new THREE.Mesh(
        new THREE.BoxGeometry(0.18, 0.08, 0.04),
        new THREE.MeshStandardMaterial({ color: 0xc0413a, roughness: 0.5 })
    );
    r.position.set(x, 0.9, -1.84);
    cart.add(r);
}
tail(-0.75);
tail(0.75);

// ---- orient cart: forward should be -Z (away from camera) ----
cart.position.set(0, 0, 0);
cart.rotation.y = Math.PI;
scene.add(cart);

// Soft contact shadow beneath cart
const contact = new THREE.Mesh(
    new THREE.CircleGeometry(2.0, 32),
    new THREE.MeshBasicMaterial({
        color: 0x000000,
        transparent: true,
        opacity: 0.1
    })
);
contact.rotation.x = -Math.PI / 2;
contact.position.y = 0.005;
scene.add(contact);

// ---------- Animate ----------
// window.__mph is updated by pollState() in the page script; we read
// it here without coupling the two files. Units: world-units per second
// per MPH. 0.45 puts 20 MPH at one lane dash (~4.8 units) every ~0.55 s,
// which reads as a relaxed, matches-the-camera-height cruise.
const SPEED_WORLD_PER_MPH = 0.79;

let t = 0;
let prevFrame = performance.now();
function animate(now) {
    if (now === undefined) now = performance.now();
    const dt = Math.min(0.05, (now - prevFrame) / 1000);
    prevFrame = now;
    t += dt;
    cart.position.y = Math.sin(t * 2.0) * 0.008;

    updatePaths(Number(window.__steerDeg) || 0, dt);

    const mph = Math.max(0, Number(window.__mph) || 0);
    if (mph > 0) {
        const slide = mph * SPEED_WORLD_PER_MPH * dt;
        // Each lane has laneDashes.length / 2 dashes tiled at DASH_STRIDE
        // spacing, so that product is the per-lane repeat distance.
        const wrap = (laneDashes.length / 2) * DASH_STRIDE;
        for (const m of laneDashes) {
            m.position.z += slide;
            if (m.position.z > DASH_Z_NEAR) {
                m.position.z -= wrap;
            }
        }
    }

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}
animate();

// ---------- Resize ----------
window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
