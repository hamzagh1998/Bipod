import { context, build } from "esbuild";
import process from "node:process";

const watch = process.argv.includes("--watch");

const common = {
  entryPoints: ["frontend/js/coach-app.jsx"],
  bundle: true,
  outfile: "frontend/js/coach-app.bundle.js",
  platform: "browser",
  format: "iife",
  target: ["es2020"],
  jsx: "automatic",
  loader: {
    ".js": "jsx",
    ".jsx": "jsx"
  },
  sourcemap: watch ? "inline" : false,
  minify: !watch,
  legalComments: "none"
};

if (watch) {
  const ctx = await context(common);
  await ctx.watch();
  console.log("[coach-build] watching frontend/js/coach-app.jsx");
} else {
  await build(common);
  console.log("[coach-build] built frontend/js/coach-app.bundle.js");
}
