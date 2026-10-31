// Render labelled evaluation data with the pinned public browser.
// Keep this command separate from benchmark_max.py, which reads only RGB/calibration.
import fs from "node:fs/promises";
import path from "node:path";
import { createServer } from "node:http";
import { pathToFileURL } from "node:url";
const [runtimePath, requestPath, outputPath, dataset] = process.argv.slice(2);
if (!outputPath)
  throw Error(
    "Usage: node render_benchmark.mjs RUNTIME.json REQUESTS.json OUTPUT_DIR [DATASET]",
  );
const runtime = JSON.parse(await fs.readFile(runtimePath, "utf8"));
const repo = runtime.source,
  vendor = runtime.three;
const { chromium } = await import(
  pathToFileURL(path.join(path.dirname(vendor), "playwright/index.mjs")).href
);
const injection = `
window.renderDocument=async(document)=>{
 holdJson.value=JSON.stringify(document);loadSceneFromJson();
 if(holdStatus.dataset.error==='true')throw Error(holdStatus.textContent);
 await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
 renderStereoView();return await captureStereoPng();
};`;
const server = createServer(async (req, res) => {
  try {
    const relative = decodeURIComponent(
      new URL(req.url, "http://localhost").pathname,
    );
    if (relative.split("/").includes("..")) throw Error("Invalid path");
    const file = relative.startsWith("/vendor/three/")
      ? path.join(vendor, relative.slice(14))
      : path.join(
          repo,
          relative.endsWith("/") ? relative + "index.html" : relative,
        );
    let bytes = await fs.readFile(file);
    if (file.endsWith("index.html"))
      bytes = Buffer.from(
        bytes
          .toString()
          .replaceAll(
            "https://cdn.jsdelivr.net/npm/three@0.180.0/",
            "/vendor/three/",
          ),
      );
    if (file.endsWith("main.js"))
      bytes = Buffer.from(bytes.toString() + injection);
    res
      .writeHead(200, {
        "Content-Type": file.endsWith(".js")
          ? "text/javascript"
          : file.endsWith(".html")
            ? "text/html"
            : file.endsWith(".css")
              ? "text/css"
              : "application/octet-stream",
      })
      .end(bytes);
  } catch (error) {
    res.writeHead(404).end(String(error));
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({
  headless: true,
  args: [
    "--enable-webgl",
    "--enable-unsafe-swiftshader",
    "--use-gl=angle",
    "--use-angle=swiftshader",
  ],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/ThreeJS/`);
  await page.addStyleTag({
    content:
      "#stereo-scene{width:1398px!important;height:318px!important;box-sizing:content-box!important;border:0!important;}",
  });
  await page.evaluate(() => window.dispatchEvent(new Event("resize")));
  await page.waitForFunction(() => window.RCRWallAPI?.isReady(), null, {
    timeout: 120000,
  });
  const document = JSON.parse(await fs.readFile(requestPath, "utf8"));
  const selected = dataset ? document[dataset] : document;
  const requests = Array.isArray(selected)
    ? selected
    : selected.cases.map((item) => ({
        ...selected.defaults,
        ...item,
        options: { ...selected.defaults.options, ...item.options },
      }));
  const results = [];
  for (const request of requests) {
    {
      if (!/^[A-Za-z0-9_-]+$/.test(request.id)) throw Error("Invalid case ID");
      const folder = path.join(outputPath, request.id);
      await fs.mkdir(folder, { recursive: true });
      let generated = await page.evaluate(
        async (r) => await window.RCRWallAPI.generateRandomWall(r),
        request.options,
      );
      if (request.scene) {
        generated = {
          stereoPngDataUrl: await page.evaluate(
            (doc) => window.renderDocument(doc),
            request.scene,
          ),
          sceneJson: JSON.stringify(request.scene),
        };
      }
      await fs.writeFile(
        path.join(folder, "stereo.png"),
        Buffer.from(generated.stereoPngDataUrl.split(",")[1], "base64"),
      );
      await fs.writeFile(
        path.join(folder, "expected_scene.json"),
        generated.sceneJson,
      );
      await fs.writeFile(
        path.join(folder, "expected_holds.json"),
        JSON.stringify(JSON.parse(generated.sceneJson).holds, null, 2),
      );
      await fs.writeFile(
        path.join(folder, "calibration.json"),
        JSON.stringify(request.calibration, null, 2),
      );
      results.push({ id: request.id, split: "validation" });
    }
    console.log(JSON.stringify({ id: request.id, completed: true }));
  }
  const target = path.join(outputPath, "manifest.json");
  await fs.writeFile(
    target,
    JSON.stringify(
      {
        cases: results,
        scope:
          "Public-source evaluation scenes. Inference reads RGB and declared calibration only.",
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
  server.close();
}
