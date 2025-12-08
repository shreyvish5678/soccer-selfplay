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

const SIZE = Math.sqrt(WIDTH ** 2 + HEIGHT ** 2);
const MAX_STEPS = 5000;

const ACTION_VECTORS = [
    [0, 0],   
    [0, -1],  
    [0, 1],  
    [-1, 0], 
    [1, 0],  
    [-1, -1],
    [1, -1],
    [-1, 1],  
    [1, 1], 
].map(v => [v[0] * PLAYER_ACCELERATION, v[1] * PLAYER_ACCELERATION]);

let player1Pos = [0, 0];
let player2Pos = [0, 0];
let ballPos = [0, 0];
let player1Vel = [0, 0];
let player2Vel = [0, 0];
let ballVel = [0, 0];
let scoreP1 = 0;
let scoreP2 = 0;
let currentStep = 0;

let session = null;
let player1Type = 'AI';
let player2Type = 'AI';

let canvas, ctx;
let animationId = null;

const keys = {};

async function init() {
    try {
        session = await ort.InferenceSession.create('soccer_ai.onnx');
        console.log('Model loaded successfully');
    } catch (e) {
        console.error('Failed to load model:', e);
        alert('Failed to load AI model. Please ensure soccer_ai.onnx is in the web_demo folder.');
    }
    
    document.getElementById('start-button').addEventListener('click', startGame);
    document.getElementById('back-button').addEventListener('click', backToMenu);
    
    window.addEventListener('keydown', e => {
        keys[e.key.toLowerCase()] = true;
        if (e.key === 'Escape') backToMenu();
    });
    window.addEventListener('keyup', e => keys[e.key.toLowerCase()] = false);
}

function startGame() {
    player1Type = document.querySelector('input[name="player1"]:checked').value;
    player2Type = document.querySelector('input[name="player2"]:checked').value;
    
    canvas = document.getElementById('game-canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    ctx = canvas.getContext('2d');
    
    document.getElementById('menu-screen').style.display = 'none';
    document.getElementById('game-screen').style.display = 'flex';
    
    resetGame();
    
    gameLoop();
}

function backToMenu() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    document.getElementById('game-screen').style.display = 'none';
    document.getElementById('menu-screen').style.display = 'flex';
}

function resetGame() {
    resetPositions();
    scoreP1 = 0;
    scoreP2 = 0;
    currentStep = 0;
    updateScoreDisplay();
}

function resetPositions() {
    player1Pos = [OFFSET, HEIGHT / 2];
    player2Pos = [WIDTH - OFFSET, HEIGHT / 2];
    ballPos = [WIDTH / 2, HEIGHT / 2];
    player1Vel = [0, 0];
    player2Vel = [0, 0];
    ballVel = [0, 0];
}

function getState() {
    const distP1Ball = norm([ballPos[0] - player1Pos[0], ballPos[1] - player1Pos[1]]) / SIZE;
    const distP2Ball = norm([ballPos[0] - player2Pos[0], ballPos[1] - player2Pos[1]]) / SIZE;
    
    return [
        player1Pos[0] / WIDTH, player1Pos[1] / HEIGHT,
        player1Vel[0] / PLAYER_MAX_SPEED, player1Vel[1] / PLAYER_MAX_SPEED,
        player2Pos[0] / WIDTH, player2Pos[1] / HEIGHT,
        player2Vel[0] / PLAYER_MAX_SPEED, player2Vel[1] / PLAYER_MAX_SPEED,
        ballPos[0] / WIDTH, ballPos[1] / HEIGHT,
        ballVel[0] / BALL_MAX_SPEED, ballVel[1] / BALL_MAX_SPEED,
        (ballPos[0] - player1Pos[0]) / WIDTH, (ballPos[1] - player1Pos[1]) / HEIGHT,
        (ballPos[0] - player2Pos[0]) / WIDTH, (ballPos[1] - player2Pos[1]) / HEIGHT,
        (player2Pos[0] - player1Pos[0]) / WIDTH, (player2Pos[1] - player1Pos[1]) / HEIGHT,
        (ballPos[0] - GOAL_WIDTH) / WIDTH,
        (WIDTH - ballPos[0] - GOAL_WIDTH) / WIDTH,
        ballPos[1] / HEIGHT - 0.5,
        distP1Ball > distP2Ball ? 1 : 0
    ];
}

async function getAIAction(playerSide) {
    if (!session) return 0;
    
    const state = getState();
    state.push(playerSide);  
    try {
        const input = new ort.Tensor('float32', new Float32Array(state), [1, 23]);
        const feeds = { state: input };
        const results = await session.run(feeds);
        const logits = results.logits.data;
        
        const maxLogit = Math.max(...logits);
        const expLogits = Array.from(logits).map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(x => x / sumExp);
        
        const rand = Math.random();
        let cumSum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) return i;
        }
        return probs.length - 1;
    } catch (e) {
        console.error('AI inference error:', e);
        return 0;
    }
}

function getHumanAction(isP1) {
    const keymap = isP1 
        ? { up: 'w', down: 's', left: 'a', right: 'd' }
        : { up: 'arrowup', down: 'arrowdown', left: 'arrowleft', right: 'arrowright' };
    
    const x = (keys[keymap.right] ? 1 : 0) - (keys[keymap.left] ? 1 : 0);
    const y = (keys[keymap.down] ? 1 : 0) - (keys[keymap.up] ? 1 : 0);
    
    const dirToAction = {
        '0,-1': 1, '0,1': 2, '-1,0': 3, '1,0': 4,
        '-1,-1': 5, '1,-1': 6, '-1,1': 7, '1,1': 8
    };
    return dirToAction[`${x},${y}`] || 0;
}

async function gameLoop() {
    let actionP1, actionP2;
    if (player1Type === 'HUMAN') {
        actionP1 = getHumanAction(true);
    } else {
        actionP1 = await getAIAction(0.0);
    }
    
    if (player2Type === 'HUMAN') {
        actionP2 = getHumanAction(false);
    } else {
        actionP2 = await getAIAction(1.0);
    }
    
    const p1Acc = ACTION_VECTORS[actionP1];
    const p2Acc = ACTION_VECTORS[actionP2];
    
    updatePhysics(player1Pos, player1Vel, p1Acc, PLAYER_MAX_SPEED, PLAYER_FRICTION);
    updatePhysics(player2Pos, player2Vel, p2Acc, PLAYER_MAX_SPEED, PLAYER_FRICTION);
    
    clampPos(player1Pos, player1Vel, PLAYER_SIZE);
    clampPos(player2Pos, player2Vel, PLAYER_SIZE);
    
    for (let i = 0; i < 2; i++) {
        resolveCollisions(player1Pos, player1Vel, PLAYER_MASS, PLAYER_SIZE, ballPos, ballVel, BALL_MASS, BALL_SIZE, PLAYER_RESTITUTION);
        resolveCollisions(player2Pos, player2Vel, PLAYER_MASS, PLAYER_SIZE, ballPos, ballVel, BALL_MASS, BALL_SIZE, PLAYER_RESTITUTION);
    }
    
    ballPos[0] += ballVel[0];
    ballPos[1] += ballVel[1];
    
    handleWallCollisions();
    applyCornerRepulsion();
    
    const ballSpeed = norm(ballVel);
    if (ballSpeed > BALL_MAX_SPEED) {
        ballVel[0] = (ballVel[0] / ballSpeed) * BALL_MAX_SPEED;
        ballVel[1] = (ballVel[1] / ballSpeed) * BALL_MAX_SPEED;
    }
    ballVel[0] *= BALL_FRICTION;
    ballVel[1] *= BALL_FRICTION;
    
    const goalCode = checkGoal();
    if (goalCode > 0) {
        resetPositions();
        updateScoreDisplay();
    }
    
    currentStep++;
    if (currentStep >= MAX_STEPS) {
        resetGame();
    }
    
    render();
    
    animationId = requestAnimationFrame(gameLoop);
}

function updatePhysics(pos, vel, acc, maxSpeed, friction) {
    vel[0] += acc[0];
    vel[1] += acc[1];
    
    const speed = norm(vel);
    if (speed > maxSpeed) {
        vel[0] = (vel[0] / speed) * maxSpeed;
        vel[1] = (vel[1] / speed) * maxSpeed;
    }
    
    if (acc[0] === 0) vel[0] *= friction;
    if (acc[1] === 0) vel[1] *= friction;
    
    pos[0] += vel[0];
    pos[1] += vel[1];
}

function clampPos(pos, vel, radius) {
    if (pos[0] < radius) {
        pos[0] = radius;
        vel[0] = 0;
    }
    if (pos[0] > WIDTH - radius) {
        pos[0] = WIDTH - radius;
        vel[0] = 0;
    }
    if (pos[1] < radius) {
        pos[1] = radius;
        vel[1] = 0;
    }
    if (pos[1] > HEIGHT - radius) {
        pos[1] = HEIGHT - radius;
        vel[1] = 0;
    }
}

function resolveCollisions(pos1, vel1, mass1, size1, pos2, vel2, mass2, size2, restitution) {
    const deltaPos = [pos2[0] - pos1[0], pos2[1] - pos1[1]];
    const distance = norm(deltaPos);
    
    if (distance >= size1 + size2) return;
    
    const distSafe = Math.max(distance, 1e-6);
    const collisionNormal = [deltaPos[0] / distSafe, deltaPos[1] / distSafe];
    const overlap = size1 + size2 - distance;
    
    const mass1Ratio = mass2 / (mass1 + mass2);
    const mass2Ratio = mass1 / (mass1 + mass2);
    
    pos1[0] -= collisionNormal[0] * overlap * mass1Ratio;
    pos1[1] -= collisionNormal[1] * overlap * mass1Ratio;
    pos2[0] += collisionNormal[0] * overlap * mass2Ratio;
    pos2[1] += collisionNormal[1] * overlap * mass2Ratio;
    
    const relVel = [vel2[0] - vel1[0], vel2[1] - vel1[1]];
    const velAlongNormal = relVel[0] * collisionNormal[0] + relVel[1] * collisionNormal[1];
    
    if (velAlongNormal >= 0) return;
    
    let impulseMag = -(1 + restitution) * velAlongNormal;
    impulseMag /= (1 / mass1 + 1 / mass2);
    
    const impulse = [impulseMag * collisionNormal[0], impulseMag * collisionNormal[1]];
    
    vel1[0] -= impulse[0] / mass1;
    vel1[1] -= impulse[1] / mass1;
    vel2[0] += impulse[0] / mass2;
    vel2[1] += impulse[1] / mass2;
    
    if (mass2 === BALL_MASS) {
        vel2[0] += (Math.random() - 0.5) * 2 * RANDOM_IMPULSE_NOISE_BALL;
        vel2[1] += (Math.random() - 0.5) * 2 * RANDOM_IMPULSE_NOISE_BALL;
    }
}

function handleWallCollisions() {
    const topGoal = (HEIGHT - GOAL_HEIGHT) / 2;
    const bottomGoal = topGoal + GOAL_HEIGHT;
    const inGoalY = ballPos[1] > topGoal && ballPos[1] < bottomGoal;
    
    if (ballPos[0] - BALL_SIZE < GOAL_WIDTH && !inGoalY) {
        ballPos[0] = GOAL_WIDTH + BALL_SIZE;
        ballVel[0] = Math.abs(ballVel[0]) * WALL_BOUNCE;
    }
    if (ballPos[0] + BALL_SIZE > WIDTH - GOAL_WIDTH && !inGoalY) {
        ballPos[0] = WIDTH - GOAL_WIDTH - BALL_SIZE;
        ballVel[0] = -Math.abs(ballVel[0]) * WALL_BOUNCE;
    }
    if (ballPos[1] - BALL_SIZE < 0) {
        ballPos[1] = BALL_SIZE;
        ballVel[1] = Math.abs(ballVel[1]) * WALL_BOUNCE;
    }
    if (ballPos[1] + BALL_SIZE > HEIGHT) {
        ballPos[1] = HEIGHT - BALL_SIZE;
        ballVel[1] = -Math.abs(ballVel[1]) * WALL_BOUNCE;
    }
}

function applyCornerRepulsion() {
    const corners = [[0, 0], [WIDTH, 0], [0, HEIGHT], [WIDTH, HEIGHT]];
    let forceX = 0, forceY = 0;
    
    for (const corner of corners) {
        const diff = [ballPos[0] - corner[0], ballPos[1] - corner[1]];
        const dist = norm(diff);
        
        if (dist < CORNER_RADIUS) {
            const safeDist = Math.max(dist, 1e-5);
            forceX += (diff[0] / safeDist);
            forceY += (diff[1] / safeDist);
        }
    }
    
    ballVel[0] += forceX * CORNER_FORCE;
    ballVel[1] += forceY * CORNER_FORCE;
}

function checkGoal() {
    const topGoal = (HEIGHT - GOAL_HEIGHT) / 2;
    const bottomGoal = topGoal + GOAL_HEIGHT;
    const inGoalY = ballPos[1] > topGoal && ballPos[1] < bottomGoal;
    
    if (ballPos[0] - BALL_SIZE <= GOAL_WIDTH && inGoalY) {
        scoreP2++;
        return 2;
    }
    if (ballPos[0] + BALL_SIZE >= WIDTH - GOAL_WIDTH && inGoalY) {
        scoreP1++;
        return 1;
    }
    return 0;
}

function updateScoreDisplay() {
    document.getElementById('score-p1').textContent = scoreP1;
    document.getElementById('score-p2').textContent = scoreP2;
}

function render() {
    ctx.fillStyle = '#468C46';
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    
    const topGoal = (HEIGHT - GOAL_HEIGHT) / 2;
    
    ctx.fillStyle = '#C80000';
    ctx.fillRect(0, topGoal, GOAL_WIDTH, GOAL_HEIGHT);
    ctx.fillStyle = '#0000C8';
    ctx.fillRect(WIDTH - GOAL_WIDTH, topGoal, GOAL_WIDTH, GOAL_HEIGHT);
    
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(WIDTH / 2, 0);
    ctx.lineTo(WIDTH / 2, HEIGHT);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(WIDTH / 2, HEIGHT / 2, 70, 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.fillStyle = '#C80000';
    ctx.beginPath();
    ctx.arc(player1Pos[0], player1Pos[1], PLAYER_SIZE, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = '#0000C8';
    ctx.beginPath();
    ctx.arc(player2Pos[0], player2Pos[1], PLAYER_SIZE, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(ballPos[0], ballPos[1], BALL_SIZE, 0, Math.PI * 2);
    ctx.fill();
}

function norm(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1]);
}

init();