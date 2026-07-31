import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the DiatomCascadeNet research site", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>DiatomCascadeNet \| Audited Research Pipeline<\/title>/i);
  assert.match(html, /DiatomCascadeNet/);
  assert.match(html, /arxiv\.org\/abs\/2512\.06613/);
  assert.match(html, /github\.com\/DinaberryPi\/DiatomCascadeNet-public/);
  assert.match(html, /Naviculoid_diatom\.jpg/);
  assert.doesNotMatch(html, /codex-preview|loading skeleton/i);
});

test("keeps unpublished inference and private data out of the site", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /层级硅藻图像分类/);
  assert.match(page, /CC BY-SA 4\.0/);
  assert.doesNotMatch(page, /type=["']file["']/i);
  assert.doesNotMatch(page, /live classifier|upload (?:an? )?image|prediction API/i);
  assert.doesNotMatch(page, /Beam search|Krammer|What the audit changes|本次审计改变了什么|quarantined/i);
  assert.doesNotMatch(page, /(?:dataset|outputs)[\\/]raw/i);
});
