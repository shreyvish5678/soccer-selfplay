const WIDTH = 600;
const HEIGHT = 400;
const GOAL_WIDTH = 10;
const GOAL_HEIGHT = 150;
const PLAYER_SIZE = 15;
const BALL_SIZE = 10;
const OFFSET = 150;

const PLAYER_ACCELERATION = 0.5;
const PLAYER_MAX_SPEED = 6.0;
const PLAYER_FRICTION = 0.9;

const BALL_FRICTION = 0.98;
const BALL_MAX_SPEED = 15.0;
const WALL_BOUNCE = 0.8;

const PLAYER_RESTITUTION = 0.5;
const BALL_RESTITUTION = 0.7;
const RANDOM_IMPULSE_NOISE_BALL = 1.5;

const PLAYER_MASS = 2.0;
const BALL_MASS = 1.0;

const CORNER_RADIUS = 30.0;
const CORNER_FORCE = 1.5;

const SIZE = Math.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT);

const actionVectors = [
    [0, 0],
    [0, -1], [0, 1], [-1, 0], [1, 0],
    [-1, -1], [1, -1], [-1, 1], [1, 1]
].map(v => [v[0] * PLAYER_ACCELERATION, v[1] * PLAYER_ACCELERATION]);

let canvas, ctx;
let p1Type, p2Type;
let session = null;
let keys = {};

let p1Pos = [OFFSET, HEIGHT / 2];
let p2Pos = [WIDTH - OFFSET, HEIGHT / 2];
let ballPos = [WIDTH / 2, HEIGHT / 2];
let p1Vel = [0, 0];
let p2Vel = [0, 0];
let ballVel = [0, 0];

let scoreP1 = 0;
let scoreP2 = 0;

let prevDistP1 = null;
let prevDistP2 = null;
let prevBallDistLeft = null;
let prevBallDistRight = null;

document.getElementById('play-btn').addEventListener('click', startGame);
document.getElementById('back-btn').addEventListener('click', () => location.reload());

window.addEventListener('keydown', e => keys[e.key.toLowerCase()] = true);
window.addEventListener('keyup', e => keys[e.key.toLowerCase()] = false);

async function startGame() {
    p1Type = document.querySelector('input[name="p1"]:checked').value;
    p2Type = document.querySelector('input[name="p2"]:checked').value;

    document.getElementById('menu').classList.add('hidden');
    document.getElementById('game-container').classList.remove('hidden');

    canvas = document.getElementById('game-canvas');
    ctx = canvas.getContext('2d');

    try {
        session = await ort.InferenceSession.create('./soccer_ai.onnx', {
            executionProviders: ['wasm', 'webgl']
        });
    } catch (e) {
        alert("Failed to load AI model. Check console.");
        console.error(e);
    }

    resetGame();
    requestAnimationFrame(gameLoop);
}

function resetGame() {
    p1Pos = [OFFSET, HEIGHT / 2];
    p2Pos = [WIDTH - OFFSET, HEIGHT / 2];
    ballPos = [WIDTH / 2, HEIGHT / 2];
    p1Vel = [0, 0];
    p2Vel = [0, 0];
    ballVel = [0, 0];

    const ballToP1 = distance(ballPos, p1Pos);
    const ballToP2 = distance(ballPos, p2Pos);
    const ballToLeft = distance(ballPos, [GOAL_WIDTH, HEIGHT / 2]);
    const ballToRight = distance(ballPos, [WIDTH - GOAL_WIDTH, HEIGHT / 2]);

    prevDistP1 = ballToP1;
    prevDistP2 = ballToP2;
    prevBallDistLeft = ballToLeft;
    prevBallDistRight = ballToRight;
}

async function getAIAction(state, isP1) {
    if (!session) return 0;
    const indicator = isP1 ? 0.0 : 1.0;
    const input = new Float32Array(state.length + 1);
    input.set(state);
    input[state.length] = indicator;

    const tensor = new ort.Tensor('float32', input, [1, input.length]);
    const feeds = { state: tensor };
    const results = await session.run(feeds);
    const logits = results.logits.data;
    const probs = softmax(logits);
    return weightedRandom(probs);
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
}

function weightedRandom(probs) {
    let r = Math.random();
    let sum = 0;
    for (let i = 0; i < probs.length; i++) {
        sum += probs[i];
        if (r <= sum) return i;
    }
    return probs.length - 1;
}

function getHumanAction(isP1) {
    const map = isP1
        ? { 'w': [0,-1], 's': [0,1], 'a': [-1,0], 'd': [1,0] }
        : { 'arrowup': [0,-1], 'arrowdown': [0,1], 'arrowleft': [-1,0], 'arrowright': [1,0] };

    let dx = 0, dy = 0;
    if (keys['w'] && isP1) dy -= 1;
    if (keys['s'] && isP1) dy += 1;
    if (keys['a'] && isP1) dx -= 1;
    if (keys['d'] && isP1) dx += 1;
    if (keys['arrowup'] && !isP1) dy -= 1;
    if (keys['arrowdown'] && !isP1) dy += 1;
    if (keys['arrowleft'] && !isP1) dx -= 1;
    if (keys['arrowright'] && !isP1) dx += 1;

    for (let i = 0; i < actionVectors.length; i++) {
        if (actionVectors[i][0] / PLAYER_ACCELERATION === dx && actionVectors[i][1] / PLAYER_ACCELERATION === dy) {
            return i;
        }
    }
    return 0;
}

function distance(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
}

function getState() {
    const normX = x => x / WIDTH;
    const normY = y => y / HEIGHT;
    const normPos = (pos) => [normX(pos[0]), normY(pos[1])];
    const normVel = (vel, maxSpeed) => [vel[0] / maxSpeed, vel[1] / maxSpeed];
    const vec = (a, b) => [(b[0] - a[0]) / WIDTH, (b[1] - a[1]) / HEIGHT];

    const distP1Ball = distance(p1Pos, ballPos);
    const distP2Ball = distance(p2Pos, ballPos);

    return [
        ...normPos(p1Pos),
        ...normVel(p1Vel, PLAYER_MAX_SPEED),
        ...normPos(p2Pos),
        ...normVel(p2Vel, PLAYER_MAX_SPEED),
        ...normPos(ballPos),
        ...normVel(ballVel, BALL_MAX_SPEED),
        ...vec(p1Pos, ballPos),
        ...vec(p2Pos, ballPos),
        ...vec(p1Pos, p2Pos),
        (ballPos[0] - GOAL_WIDTH) / WIDTH,
        (WIDTH - ballPos[0] - GOAL_WIDTH) / WIDTH,
        ballPos[1] / HEIGHT - 0.5,
        distP1Ball > distP2Ball ? 1.0 : 0.0
    ];
}

function clampPos(pos, vel, radius) {
    if (pos[0] < radius) {
        pos[0] = radius;
        if (vel[0] < 0) vel[0] = 0;
    }
    if (pos[0] > WIDTH - radius) {
        pos[0] = WIDTH - radius;
        if (vel[0] > 0) vel[0] = 0;
    }

    if (pos[1] < radius) {
        pos[1] = radius;
        if (vel[1] < 0) vel[1] = 0;
    }
    if (pos[1] > HEIGHT - radius) {
        pos[1] = HEIGHT - radius;
        if (vel[1] > 0) vel[1] = 0;
    }
}

function step(action1, action2) {
    const acc1 = actionVectors[action1];
    const acc2 = actionVectors[action2];
    p1Vel[0] += acc1[0];
    p1Vel[1] += acc1[1];
    p2Vel[0] += acc2[0];
    p2Vel[1] += acc2[1];

    const clampVel = (vel, max) => {
        const speed = Math.sqrt(vel[0]**2 + vel[1]**2);
        if (speed > max) {
            vel[0] = (vel[0] / speed) * max;
            vel[1] = (vel[1] / speed) * max;
        }
    };
    clampVel(p1Vel, PLAYER_MAX_SPEED);
    clampVel(p2Vel, PLAYER_MAX_SPEED);

    p1Pos[0] += p1Vel[0];
    p1Pos[1] += p1Vel[1];
    p2Pos[0] += p2Vel[0];
    p2Pos[1] += p2Vel[1];
    ballPos[0] += ballVel[0];
    ballPos[1] += ballVel[1];

    clampPos(p1Pos, p1Vel, PLAYER_SIZE);
    clampPos(p2Pos, p2Vel, PLAYER_SIZE);

    p1Vel[0] *= PLAYER_FRICTION;
    p1Vel[1] *= PLAYER_FRICTION;
    p2Vel[0] *= PLAYER_FRICTION;
    p2Vel[1] *= PLAYER_FRICTION;
    ballVel[0] *= BALL_FRICTION;
    ballVel[1] *= BALL_FRICTION;

    handlePlayerBallCollision();
    handleWallCollisions();
    applyCornerRepulsion();
    checkGoal();
}

function handlePlayerBallCollision() {
    const checkCollision = (playerPos, playerVel) => {
        const dx = ballPos[0] - playerPos[0];
        const dy = ballPos[1] - playerPos[1];
        const dist = Math.sqrt(dx*dx + dy*dy);
        const minDist = PLAYER_SIZE + BALL_SIZE;

        if (dist < minDist) {
            const nx = dx / dist;
            const ny = dy / dist;

            const relativeVelX = ballVel[0] - playerVel[0];
            const relativeVelY = ballVel[1] - playerVel[1];
            const dot = relativeVelX * nx + relativeVelY * ny;

            if (dot < 0) {
                const impulse = 2 * dot / (PLAYER_MASS + BALL_MASS);
                ballVel[0] -= impulse * PLAYER_MASS * nx * BALL_RESTITUTION;
                ballVel[1] -= impulse * PLAYER_MASS * ny * BALL_RESTITUTION;

                const noise = (Math.random() - 0.5) * RANDOM_IMPULSE_NOISE_BALL;
                ballVel[0] += noise;
                ballVel[1] += noise;

                const overlap = minDist - dist;
                ballPos[0] += nx * overlap * 0.5;
                ballPos[1] += ny * overlap * 0.5;
            }
        }
    };

    checkCollision(p1Pos, p1Vel);
    checkCollision(p2Pos, p2Vel);
}

function handleWallCollisions() {
    const topGoal = (HEIGHT - GOAL_HEIGHT) / 2;
    const bottomGoal = topGoal + GOAL_HEIGHT;

    if (ballPos[1] - BALL_SIZE < 0) {
        ballPos[1] = BALL_SIZE;
        ballVel[1] = Math.abs(ballVel[1]) * WALL_BOUNCE;
    }
    if (ballPos[1] + BALL_SIZE > HEIGHT) {
        ballPos[1] = HEIGHT - BALL_SIZE;
        ballVel[1] = -Math.abs(ballVel[1]) * WALL_BOUNCE;
    }

    const inGoalY = ballPos[1] > topGoal && ballPos[1] < bottomGoal;
    if (!inGoalY) {
        if (ballPos[0] - BALL_SIZE < GOAL_WIDTH) {
            ballPos[0] = GOAL_WIDTH + BALL_SIZE;
            ballVel[0] = Math.abs(ballVel[0]) * WALL_BOUNCE;
        }
        if (ballPos[0] + BALL_SIZE > WIDTH - GOAL_WIDTH) {
            ballPos[0] = WIDTH - GOAL_WIDTH - BALL_SIZE;
            ballVel[0] = -Math.abs(ballVel[0]) * WALL_BOUNCE;
        }
    }
}

function applyCornerRepulsion() {
    const corners = [[0,0], [WIDTH,0], [0,HEIGHT], [WIDTH,HEIGHT]];
    let forceX = 0, forceY = 0;

    for (const corner of corners) {
        const dx = ballPos[0] - corner[0];
        const dy = ballPos[1] - corner[1];
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < CORNER_RADIUS && dist > 0.001) {
            forceX += dx / dist * CORNER_FORCE;
            forceY += dy / dist * CORNER_FORCE;
        }
    }
    ballVel[0] += forceX;
    ballVel[1] += forceY;
}

function checkGoal() {
    const topGoal = (HEIGHT - GOAL_HEIGHT) / 2;
    const bottomGoal = topGoal + GOAL_HEIGHT;
    const inGoalY = ballPos[1] > topGoal && ballPos[1] < bottomGoal;

    if (inGoalY && ballPos[0] + BALL_SIZE >= WIDTH - GOAL_WIDTH) {
        scoreP1++;
        document.getElementById('score').textContent = `${scoreP1} - ${scoreP2}`;
        resetGame();
    }
    if (inGoalY && ballPos[0] - BALL_SIZE <= GOAL_WIDTH) {
        scoreP2++;
        document.getElementById('score').textContent = `${scoreP1} - ${scoreP2}`;
        resetGame();
    }
}

async function gameLoop() {
    const state = getState();

    const action1 = p1Type === 'ai'
        ? await getAIAction(state, true)
        : getHumanAction(true);

    const action2 = p2Type === 'ai'
        ? await getAIAction(state, false)
        : getHumanAction(false);

    step(action1, action2);
    render();

    requestAnimationFrame(gameLoop);
}

function render() {
    ctx.fillStyle = '#468847';
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    ctx.fillStyle = '#c62828';
    ctx.fillRect(0, (HEIGHT - GOAL_HEIGHT)/2, GOAL_WIDTH, GOAL_HEIGHT);
    ctx.fillStyle = '#1565c0';
    ctx.fillRect(WIDTH - GOAL_WIDTH, (HEIGHT - GOAL_HEIGHT)/2, GOAL_WIDTH, GOAL_HEIGHT);

    ctx.strokeStyle = 'white';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(WIDTH/2, 0);
    ctx.lineTo(WIDTH/2, HEIGHT);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(WIDTH/2, HEIGHT/2, 70, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = '#c62828';
    ctx.beginPath();
    ctx.arc(p1Pos[0], p1Pos[1], PLAYER_SIZE, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#1565c0';
    ctx.beginPath();
    ctx.arc(p2Pos[0], p2Pos[1], PLAYER_SIZE, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(ballPos[0], ballPos[1], BALL_SIZE, 0, Math.PI * 2);
    ctx.fill();
}