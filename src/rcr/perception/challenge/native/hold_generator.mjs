// Invoke unchanged geometry/sampling functions from the pinned public source.
// Node supplies Three.js math and mesh parsing; no browser or GPU is needed.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";

const [repo, requestFile, outputFile, threeRoot] = process.argv.slice(2);
const load = (relative) =>
  import(pathToFileURL(path.join(threeRoot, relative)).href);
const THREE = await load("build/three.module.js");
const { ConvexGeometry } = await load(
  "examples/jsm/geometries/ConvexGeometry.js",
);
const { STLLoader } = await load("examples/jsm/loaders/STLLoader.js");
const source = fs.readFileSync(path.join(repo, "ThreeJS/main.js"), "utf8");
const hash = crypto.createHash("sha256").update(source).digest("hex");
if (hash !== "54240e2ba03fb9a8c3ddf13e1458d2110a8140a8db899ce0a93dd3d8d0fb1a4b")
  throw Error("Unvalidated generator source");
const names = [
  "seededRandom",
  "constrainVolumeSize",
  "constrainVolumeDepth",
  "volumeBaseTriangle",
  "volumeBasesOverlap",
  "createRandomSceneSpec",
  "makeTriangleSurface",
  "addPyramidVolume",
  "makeHold",
  "getHoldTypeFromRecordType",
  "normalizeHoldGeometry",
  "createHoldHull",
  "transformHoldHull",
  "holdHullsIntersect",
  "createVolumeHull",
  "sampleSurface",
  "addHolds",
  "serializeSceneData",
];
const functions = names
  .map((name) => {
    const match = source.match(
      new RegExp("^function " + name + "\\([\\s\\S]*?^}", "m"),
    );
    if (!match) throw Error("Missing source function: " + name);
    return match[0];
  })
  .join("\n");
const geometry = {},
  hulls = {};
const build = new Function(
  "THREE",
  "ConvexGeometry",
  "structuredClone",
  "holdGeometries",
  "holdHulls",
  "holdRecordTypes",
  source.slice(
    source.indexOf("const WALL_WIDTH"),
    source.indexOf("const canvas ="),
  ) +
    "\n" +
    "let currentSeed=0,currentMaterialSeed=0,currentSceneSpec=null;\n" +
    functions +
    "\nreturn {" +
    names.join(",") +
    ",setState(s,m,g){currentSeed=s;currentMaterialSeed=m;currentSceneSpec=g;}};",
);
const context = build(THREE, ConvexGeometry, structuredClone, geometry, hulls, {
  Microban: "microban_hold",
  CubeHold: "cube_hold",
  Sidepull: "sidepull_hold",
  Fingerpocket: "fingerpocket_hold",
});

for (const [name, file] of Object.entries({
  Microban: "microban_hold",
  CubeHold: "cube_hold",
  Sidepull: "sidehpull_hold",
  Fingerpocket: "fingerpocket_hold",
})) {
  const bytes = fs.readFileSync(path.join(repo, "Assets", file + ".stl"));
  geometry[name] = context.normalizeHoldGeometry(
    new STLLoader().parse(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    ),
  );
  hulls[name] = context.createHoldHull(geometry[name]);
}
const colors = ["#F2554E", "#59C77B", "#6389ED"];
const types = ["microban_hold", "sidepull_hold", "fingerpocket_hold"];
const results = [];
const requests = JSON.parse(fs.readFileSync(requestFile, "utf8"));
for (const request of requests) {
  const { seed, volume_count, hold_count, width, height } = request;
  const group = new THREE.Group(),
    surfaces = [];
  let spec;
  try {
    spec = context.createRandomSceneSpec(seed, volume_count, width, height);
    if (
      request.estimated_volumes?.length &&
      !request.estimated_volumes.every((v) =>
        spec.volumes.some(
          (w) =>
            Math.hypot(v[0] - w.center.x, v[1] - w.center.y) < 0.025 &&
            Math.abs(v[2] - w.baseWidth) < 0.035 &&
            Math.abs(v[3] - w.depth) < 0.025,
        ),
      )
    )
      continue;
    const p = (x, y, z = 0) => new THREE.Vector3(x, y, z);
    surfaces.push({
      points: [p(-width / 2, 0), p(width / 2, 0), p(width / 2, height)],
      normal: p(0, 0, 1),
      area: (width * height) / 2,
    });
    surfaces.push({
      points: [p(-width / 2, 0), p(width / 2, height), p(-width / 2, height)],
      normal: p(0, 0, 1),
      area: (width * height) / 2,
    });
    const material = new THREE.MeshBasicMaterial();
    for (const volume of spec.volumes)
      context.addPyramidVolume(group, surfaces, volume, spec.wall, material);
    // Conservative prefilter: the first 120 proposals include rejected draws.
    // A true scene still requires exact hull rejection and image verification.
    if (request.estimate) {
      const rand = context.seededRandom(seed ^ 0x9e3779b9),
        matched = new Set();
      const estimates = Object.values(request.estimate);
      for (let i = 0; i < 120; i++) {
        const color = colors[Math.floor(rand() * 3)],
          type = types[Math.floor(rand() * 3)];
        const { point, normal, volumeName } = context.sampleSurface(
          surfaces,
          rand,
        );
        rand();
        if (volumeName) point.addScaledVector(normal, Math.sqrt(3) * 0.001);
        for (let j = 0; j < estimates.length; j++) {
          const h = estimates[j];
          if (
            h.color.toUpperCase() === color &&
            h.type === type &&
            Math.hypot(
              h.position.x - point.x,
              h.position.y - point.y,
              h.position.z - point.z,
            ) < 0.006
          )
            matched.add(j);
        }
      }
      if (matched.size < Math.max(6, Math.ceil(estimates.length * 0.45)))
        continue;
    }
    const holds = context.addHolds(
      group,
      surfaces,
      hold_count,
      seed,
      spec.volumes.map(context.createVolumeHull),
    );
    group.updateMatrixWorld(true);
    context.setState(seed, (seed ^ 0x9e3779b9) >>> 0, spec);
    results.push({
      id: request.id,
      seed,
      volume_count,
      scene: JSON.parse(context.serializeSceneData(holds)),
    });
    fs.appendFileSync(
      outputFile + ".jsonl",
      JSON.stringify(results.at(-1)) + "\n",
    );
  } catch (error) {
    results.push({ id: request.id, seed, volume_count, error: String(error) });
  } finally {
    group.traverse((object) => {
      if (object.geometry && !Object.values(geometry).includes(object.geometry))
        object.geometry.dispose();
      if (object.material) object.material.dispose();
    });
  }
}
fs.writeFileSync(
  outputFile,
  JSON.stringify({ source_sha256: hash, requests: requests.length, results }),
);
console.log(
  JSON.stringify({
    requests: requests.length,
    generated: results.filter((r) => r.scene).length,
    errors: results.filter((r) => r.error).length,
  }),
);
