import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const {
  getHealthMock,
  validateImageMock,
  predictImageMock,
  createThumbnailMock,
} = vi.hoisted(() => ({
  getHealthMock: vi.fn(),
  validateImageMock: vi.fn().mockResolvedValue(
    "The image passed file-format checks.",
  ),
  predictImageMock: vi.fn(),
  createThumbnailMock: vi.fn().mockResolvedValue("data:image/jpeg;base64,thumb"),
}));

vi.mock("./thumbnail", () => ({
  createThumbnail: createThumbnailMock,
}));

vi.mock("./api/client", () => ({
  getHealth: getHealthMock,
  predictImage: predictImageMock,
  validateImage: validateImageMock,
}));

describe("dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    getHealthMock.mockResolvedValue({
      status: "ok",
      predictions_available: false,
      model_message: "No trained research model is configured.",
    });
  });

  it("shows only upload, prediction, confidence, and a short research disclaimer", async () => {
    render(<App />);
    expect(
      await screen.findByRole("heading", { name: /Glaucoma screening/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /Upload image/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Prediction" })).toBeInTheDocument();
    expect(
      screen.getByText(/Research use only\. Not for diagnosis or treatment\./i),
    ).toBeInTheDocument();
  });

  it("hides API status, metrics, and charts", async () => {
    render(<App />);
    await screen.findByRole("heading", { name: /Glaucoma screening/i });

    for (const hiddenText of [
      "API online",
      "Model status:",
      "MODEL EVALUATION",
      "Evaluation Metrics",
      "Accuracy",
      "Precision",
      "Recall",
      "F1-Score",
      "Specificity",
      "Confusion Matrix",
      "Training and validation loss",
      "Training and validation accuracy",
      "ROC curve",
      "Model details",
      "API documentation",
    ]) {
      expect(screen.queryByText(hiddenText)).not.toBeInTheDocument();
    }
  });

  it("does not claim to predict when the model is unavailable", async () => {
    render(<App />);
    const user = userEvent.setup();
    const file = new File(["fundus"], "fundus.webp", { type: "image/webp" });

    await user.upload(screen.getByLabelText(/choose an image/i), file);

    expect(screen.getByRole("button", { name: /analyze image/i })).toBeDisabled();
    expect(validateImageMock).not.toHaveBeenCalled();
    expect(predictImageMock).not.toHaveBeenCalled();
  });

  it("shows the prediction and confidence after image analysis", async () => {
    getHealthMock.mockResolvedValueOnce({
      status: "ok",
      predictions_available: true,
      model_message: "Model available.",
    });
    predictImageMock.mockResolvedValueOnce({
      prediction: "glaucoma",
      confidence: 0.874,
      glaucoma_probability: 0.874,
      threshold: 0.5,
      threshold_source: "checkpoint",
      disclaimer: "Research use only.",
      model_name: "resnet50",
      explanation_method: "Grad-CAM",
      explanation_note: "",
      explanation_error: "",
      original_image: null,
      gradcam_heatmap: null,
      gradcam_overlay: null,
    });
    render(<App />);
    const user = userEvent.setup();
    const file = new File(["fundus"], "fundus.jpg", { type: "image/jpeg" });

    await user.upload(await screen.findByLabelText(/choose an image/i), file);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));

    const result = await screen.findByRole("status");
    expect(result).toHaveTextContent("Yes");
    expect(result).toHaveTextContent("Confidence: 87.4%");
    expect(predictImageMock).toHaveBeenCalledWith(file);
    expect(screen.queryByText(/threshold|Grad-CAM|model details/i))
      .not.toBeInTheDocument();
    expect(await screen.findByText(/Prediction history/i)).toBeInTheDocument();
    expect(screen.getByRole("listitem")).toHaveTextContent("Yes");
    expect(screen.getByRole("listitem")).toHaveTextContent("87.4%");
    expect(JSON.parse(window.localStorage.getItem("glaucoma-screening-history") ?? "[]"))
      .toHaveLength(1);
  });

  it("toggles dark mode and persists the setting", async () => {
    render(<App />);
    const user = userEvent.setup();
    const toggle = await screen.findByRole("button", { name: "Dark mode" });

    await user.click(toggle);

    expect(screen.getByRole("button", { name: "Light mode" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(window.localStorage.getItem("glaucoma-screening-theme")).toBe("dark");
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
  });

  it("clears saved prediction history", async () => {
    window.localStorage.setItem(
      "glaucoma-screening-history",
      JSON.stringify([
        {
          id: "saved-1",
          prediction: "normal",
          confidence: 0.9,
          createdAt: "2026-11-16T12:00:00.000Z",
          thumbnail: "data:image/jpeg;base64,thumb",
        },
      ]),
    );
    render(<App />);
    const user = userEvent.setup();

    expect(await screen.findByText(/Confidence 90\.0%/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /clear history/i }));

    expect(screen.getByText(/No predictions saved yet/i)).toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem("glaucoma-screening-history") ?? "[]"))
      .toEqual([]);
  });

  it("copies the current prediction and confidence", async () => {
    getHealthMock.mockResolvedValueOnce({
      status: "ok",
      predictions_available: true,
      model_message: "Model available.",
    });
    predictImageMock.mockResolvedValueOnce({
      prediction: "normal",
      confidence: 0.91,
      glaucoma_probability: 0.09,
      threshold: 0.5,
      threshold_source: "checkpoint",
      disclaimer: "Research use only.",
      model_name: "resnet50",
      explanation_method: "Grad-CAM",
      explanation_note: "",
      explanation_error: "",
      original_image: null,
      gradcam_heatmap: null,
      gradcam_overlay: null,
    });
    render(<App />);
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const file = new File(["fundus"], "fundus.jpg", { type: "image/jpeg" });

    await user.upload(await screen.findByLabelText(/choose an image/i), file);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));
    await screen.findByText("Confidence:");
    await user.click(screen.getByRole("button", { name: /copy result/i }));

    expect(writeText).toHaveBeenCalledWith("Prediction: No\nConfidence: 91.0%");
    expect(await screen.findByText("Result copied.")).toBeInTheDocument();
  });

  it("rejects unsupported browser file types without sending a request", async () => {
    render(<App />);
    const user = userEvent.setup({ applyAccept: false });
    const file = new File(["text"], "notes.txt", { type: "text/plain" });

    await user.upload(screen.getByLabelText(/choose an image/i), file);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /choose a supported retinal fundus image/i,
    );
    expect(screen.getByRole("button", { name: /analyze image/i })).toBeDisabled();
    expect(validateImageMock).not.toHaveBeenCalled();
    expect(predictImageMock).not.toHaveBeenCalled();
  });

  it("rejects files larger than the browser upload limit", async () => {
    render(<App />);
    const user = userEvent.setup({ applyAccept: false });
    const file = new File(
      [new Uint8Array(10 * 1024 * 1024 + 1)],
      "large.png",
      { type: "image/png" },
    );

    await user.upload(screen.getByLabelText(/choose an image/i), file);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /larger than 10 MB/i,
    );
    expect(validateImageMock).not.toHaveBeenCalled();
    expect(predictImageMock).not.toHaveBeenCalled();
  });
});
