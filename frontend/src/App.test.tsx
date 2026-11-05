import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const { validateImageMock, predictImageMock, errorMessageMock } = vi.hoisted(() => ({
  validateImageMock: vi.fn().mockResolvedValue(
    "The image passed file-format checks.",
  ),
  predictImageMock: vi.fn(),
  errorMessageMock: vi.fn().mockReturnValue("Request failed."),
}));

vi.mock("./api/client", () => ({
  getHealth: vi.fn().mockResolvedValue({
    status: "ok",
    predictions_available: false,
    model_message: "No trained research model is configured.",
  }),
  getMetrics: vi.fn().mockResolvedValue({
    available: false,
    message: "No evaluation report exists yet.",
  }),
  getErrorMessage: errorMessageMock,
  predictImage: predictImageMock,
  validateImage: validateImageMock,
}));

describe("dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    render(<App />);
  });

  it("explains the research-only scope and unavailable model", async () => {
    expect(
      await screen.findByText(/No trained research model is configured/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/not a medical device and cannot diagnose glaucoma/i),
    ).toBeInTheDocument();
  });

  it("does not display fabricated evaluation metrics", async () => {
    expect(
      await screen.findByText(/No evaluation report exists yet/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Accuracy")).not.toBeInTheDocument();
  });

  it("lets a user validate an image without pretending to predict", async () => {
    const user = userEvent.setup();
    const file = new File(["fundus"], "fundus.png", { type: "image/png" });

    await user.upload(screen.getByLabelText(/choose an image/i), file);
    await user.click(
      screen.getByRole("button", { name: /validate image \(no prediction\)/i }),
    );

    expect(await screen.findByText(/passed file-format checks/i)).toBeInTheDocument();
    expect(validateImageMock).toHaveBeenCalledWith(file);
  });

  it("rejects unsupported browser file types without sending a request", async () => {
    const user = userEvent.setup({ applyAccept: false });
    const file = new File(["text"], "notes.txt", { type: "text/plain" });

    await user.upload(screen.getByLabelText(/choose an image/i), file);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /choose a JPEG or PNG/i,
    );
    expect(screen.getByRole("button", { name: /validate image/i })).toBeDisabled();
    expect(validateImageMock).not.toHaveBeenCalled();
    expect(predictImageMock).not.toHaveBeenCalled();
  });

  it("rejects files larger than the browser upload limit", async () => {
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
