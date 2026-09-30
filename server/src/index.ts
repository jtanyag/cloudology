import "dotenv/config";
import path from "node:path";
import express from "express";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import sharp from "sharp";
import rateLimit from "express-rate-limit";
import { z } from "zod";

const CLOUD_TYPES = ["Cirrus", "Cirrocumulus", "Cirrostratus", "Altocumulus", "Altostratus", "Nimbostratus", "Stratocumulus", "Stratus", "Cumulus", "Cumulonimbus", "Not a cloud"] as const;

const ResultSchema = z.object({
  cloudType: z.enum(CLOUD_TYPES),
  confidence: z.number().min(0).max(1),
  reasoning: z.string()
});

const PROMPT = `You are a meteorologist. Classify the clouds in this photo. Choose exactly one cloudType from ${CLOUD_TYPES.join(", ")}. Respond with ONLY a JSON object, no markdown:
{"cloudType: "...", "confidence": 0.0-1.0, "reasoning": "one or two sentences describing visible features"}`;

const app = express();
app.set("trust proxy", 1); // Render is behind a proxy
app.use(cors({ origin: process.env.CLIENT_ORIGIN }));
app.use("/api", rateLimit({ windowMs: 60_000, max: 10 }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 }, // 15 MB max
  fileFilter: (_req, file, cb) => cb(null, file.mimetype.startsWith("image/"))
});

app.get("/api/health", (_req, res) => res.json({ ok: true }));

app.post("/api/classify", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No image uploaded" });

  try {
    const jpeg = await sharp(req.file.buffer)
      .rotate()
      .resize({ width: 1024, height: 1024, fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 85 })
      .toBuffer();

    const dataUrl = `data:image/jpeg;base64,${jpeg.toString("base64")}`;

    const { data } = await axios.post(
      "https://router.huggingface.co/v1/chat/completions",
      {
        model: process.env.HF_MODEL,
        max_tokens: 300,
        temperature: 0.2,
        messages: [
          {
            role: "user",
            content: [
              { type: "image_url", image_url: { url: dataUrl } },
              { type: "text", text: PROMPT }
            ]
          }
        ]
      },
      {
        headers: { Authorization: `Bearer ${process.env.HF_TOKEN}` },
        timeout: 60_000
      }
    );

    const text: string = data.choices?.[0]?.message?.content ?? "";
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) {
      console.error("Model output had no JSON:", text);
      throw new Error("Model returned no JSON");
    }

    res.json(ResultSchema.parse(JSON.parse(match[0])));
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error(
        "Hugging Face request failed:",
        error.response?.status,
        JSON.stringify(error.response?.data),
        error.code
      );
    } else {
      console.error("Classification failed:", error);
    }
    const status = axios.isAxiosError(error) ? error.response?.status : undefined;
    res.status(status === 429 ? 429 : 502).json({
      error: status === 429
        ? "Model is busy, try again shortly"
        : "Could not classify the image"
    })
  }
})

// Upload errors (e.g. fil too large)
app.use(
  (error: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    res.status(400).json({ error: "Upload failed. Images must be under 15 MB." });
  }
)

// In production, serve the built React app
if (process.env.NODE_ENV === "production") {
  const clientDist = path.resolve(process.cwd(), "client/dist");
  app.use(express.static(clientDist));
  app.use((req, res, next) => {
    if (req.method === "GET" && !req.path.startsWith("/api")) {
      return res.sendFile(path.join(clientDist, "index.html"));
    }
    next();
  })
}

app.listen(process.env.PORT || 3001, () => {
  console.log(`Cloudology API listening on port ${process.env.PORT || 3001}`)
})