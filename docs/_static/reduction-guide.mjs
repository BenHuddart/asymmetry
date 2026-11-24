/* Routes describe observations, not a numerical reduction implementation. */
export const destinations = {
  asymmetry: {label: "Asymmetry over time", steps: ["bin", "ratio", "fit"]},
  fourier: {label: "Fourier spectrum", steps: ["lifetime", "fourier"]},
  maxent: {label: "Field distribution from maximum entropy", steps: ["lifetime", "maxent"]},
  integral: {label: "Scan: add counts, then take a ratio", steps: ["integral", "scan"]},
  mean: {label: "Scan: average acquisition-bin ratios", steps: ["native-mean", "scan"]},
  vector: {label: "Rotating frame from two measured components", steps: ["bin", "ratio", "vector"]},
  scalar: {label: "Slowly varying display from one signal", steps: ["bin", "ratio", "scalar"]},
  countfit: {label: "Fit to detector-group counts", steps: ["countfit"]},
};

export function traceRoute(result, deadtime, background, fourierSource) {
  const steps = ["acquire"];
  if (deadtime) steps.push("deadtime");
  steps.push("align");
  if (result !== "countfit" && background !== "none") steps.push("background");
  if (result === "fourier" && fourierSource === "ratio") {
    steps.push("bin", "ratio", "fourier");
  } else {
    steps.push(...destinations[result].steps);
  }
  const unavailable = result === "countfit" && background === "reference";
  return {steps, unavailable};
}

export function enhanceGuide(guide) {
  const panels = new Map(Array.from(guide.querySelectorAll(".reduction-step"), panel => [
    panel.id.replace("guide-", ""), panel,
  ]));
  const controls = document.createElement("div");
  controls.className = "reduction-controls";
  const selectControl = (label, name, options) => {
    const wrapper = document.createElement("label");
    wrapper.append(document.createTextNode(label));
    const select = document.createElement("select");
    select.name = name;
    for (const [value, text] of options) select.add(new Option(text, value));
    wrapper.append(select);
    controls.append(wrapper);
    return select;
  };
  const result = selectControl("Choose a result", "result",
    Object.entries(destinations).map(([key, item]) => [key, item.label]));
  const background = selectControl("Background treatment", "background", [
    ["none", "Off"], ["estimated", "Range, tail-fit, or fixed level"],
    ["reference", "Subtract a reference run"],
  ]);
  const deadtime = selectControl("Detector deadtime correction", "deadtime", [
    ["off", "Off"], ["on", "Include correction"],
  ]);
  const fourierSource = selectControl("Signal to transform", "fourier-source", [
    ["grouped", "Groups with decay envelope removed (default)"], ["ratio", "F−B asymmetry"],
  ]);
  const status = document.createElement("p");
  status.className = "reduction-status";
  status.setAttribute("role", "status");
  const overview = document.createElement("nav");
  overview.setAttribute("aria-label", "Reduction route");
  const route = document.createElement("ol");
  route.className = "reduction-route";
  overview.append(route);
  const context = document.createElement("details");
  context.className = "reduction-context";
  const otherLabel = document.createElement("summary");
  otherLabel.textContent = "Explore other steps";
  const alternatives = document.createElement("div");
  alternatives.className = "reduction-alternatives";
  alternatives.setAttribute("role", "group");
  alternatives.setAttribute("aria-label", "Steps outside this route");
  context.append(otherLabel, alternatives);
  const pager = document.createElement("div");
  pager.className = "reduction-pager";
  const previous = document.createElement("button");
  previous.type = "button";
  previous.textContent = "Previous step";
  const next = document.createElement("button");
  next.type = "button";
  next.textContent = "Next step";
  pager.append(previous, next);
  const showAll = document.createElement("button");
  showAll.type = "button";
  showAll.textContent = "Read all steps";
  showAll.setAttribute("aria-pressed", "false");
  controls.append(showAll);
  const interaction = document.createElement("div");
  interaction.className = "reduction-interaction";
  interaction.append(controls, status, overview, pager);
  guide.prepend(interaction);
  guide.append(context);
  guide.classList.add("reduction-enhanced");
  let selected = "acquire";
  let activeSteps = [];
  const buttons = new Map();
  const selectStep = (key, moveFocus = false) => {
    selected = key;
    const allVisible = showAll.getAttribute("aria-pressed") === "true";
    for (const [id, panel] of panels) panel.hidden = !allVisible && id !== key;
    for (const [id, button] of buttons) {
      if (id === key) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    }
    const index = activeSteps.indexOf(key);
    previous.disabled = index <= 0;
    next.disabled = index < 0 || index === activeSteps.length - 1;
    if (moveFocus) panels.get(key).querySelector(".rubric").focus();
  };
  const update = () => {
    const {steps, unavailable} = traceRoute(result.value, deadtime.value === "on",
      background.value, fourierSource.value);
    activeSteps = steps;
    fourierSource.parentElement.hidden = result.value !== "fourier";
    status.textContent = unavailable
      ? "This combination is unavailable: fits to detector-group counts cannot use reference-run subtraction. Choose Off or a compatible non-reference method; the model fits its own accidental-background levels."
      : `${destinations[result.value].label} — ${steps.length} steps from detector counts. Select a step to read how it works.`;
    status.classList.toggle("reduction-unavailable", unavailable);
    route.replaceChildren();
    alternatives.replaceChildren();
    buttons.clear();
    const ordered = [...steps, ...Array.from(panels.keys()).filter(key => !steps.includes(key))];
    for (const key of ordered) {
      const panel = panels.get(key);
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = panel.querySelector(".rubric").textContent;
      button.setAttribute("aria-controls", panel.id);
      button.addEventListener("click", () => selectStep(key, !steps.includes(key)));
      buttons.set(key, button);
      if (steps.includes(key)) {
        const item = document.createElement("li");
        item.append(button);
        route.append(item);
      } else {
        alternatives.append(button);
      }
    }
    selectStep(steps.includes(selected) ? selected : steps[0]);
  };
  for (const panel of panels.values()) panel.querySelector(".rubric").tabIndex = -1;
  controls.addEventListener("change", update);
  showAll.addEventListener("click", () => {
    const visible = showAll.getAttribute("aria-pressed") !== "true";
    showAll.setAttribute("aria-pressed", String(visible));
    showAll.textContent = visible ? "Read one step" : "Read all steps";
    selectStep(selected);
  });
  previous.addEventListener("click", () => selectStep(activeSteps[activeSteps.indexOf(selected) - 1], true));
  next.addEventListener("click", () => selectStep(activeSteps[activeSteps.indexOf(selected) + 1], true));
  guide.addEventListener("keydown", event => {
    const key = Array.from(buttons).find(([, button]) => button === event.target)?.[0];
    if (!key || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    const routeButtons = event.target.closest("nav");
    const sequence = Array.from(buttons)
      .filter(([, button]) => button.closest("nav") === routeButtons)
      .map(([id]) => id);
    const index = sequence.indexOf(key);
    const target = event.key === "Home" ? 0 : event.key === "End" ? sequence.length - 1
      : (index + (["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1) + sequence.length) % sequence.length;
    event.preventDefault();
    selectStep(sequence[target]);
    buttons.get(sequence[target]).focus();
  });
  update();
}

if (typeof document !== "undefined") {
  for (const guide of document.querySelectorAll(".reduction-guide")) enhanceGuide(guide);
}
