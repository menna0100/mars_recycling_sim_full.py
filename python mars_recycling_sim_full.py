
"""
MARS-CYCLE: Smart Recycling Machine Simulation (Full)
- Includes optional deep-learning classifier hook (TF/PyTorch), solar+ battery sim, ion-exchange stage, and Streamlit dashboard.
- Designed as a local proof-of-concept for a hackathon.
- Dependencies (optional): streamlit, numpy, Pillow, tensorflow OR torch
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math
import os
import sys

# Optional imports (deep model + streamlit). We'll guard imports so script can run without them.
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

try:
    # try tensorflow first
    import tensorflow as tf
    DL_BACKEND = "tensorflow"
except Exception:
    try:
        import torch
        DL_BACKEND = "torch"
    except Exception:
        DL_BACKEND = None

# ------------------------------
# Data classes
# ------------------------------
@dataclass
class WasteItem:
    id: int
    name: str
    raw_type: str  # textual label simulating ground truth
    mass_kg: float = 0.1
    image_path: Optional[str] = None  # optional image for DL classifier

@dataclass
class ProcessingResult:
    item: WasteItem
    routed_to: str
    output: Optional[str] = None
    energy_used_kwh: float = 0.0
    processing_time_s: float = 0.0

# ------------------------------
# Energy system simulation
# ------------------------------
@dataclass
class EnergySystem:
    solar_capacity_kw: float = 2.0       # peak generation capacity (kW)
    battery_capacity_kwh: float = 10.0   # total battery capacity (kWh)
    battery_soc_kwh: float = 10.0        # current stored energy (kWh)
    generation_efficiency: float = 0.9   # losses
    load_log: List[Dict] = field(default_factory=list)

    def produce_solar(self, sun_factor: float) -> float:
        """
        sun_factor: fraction [0..1] of peak (depends on time of day, dust)
        returns kWh produced over 1 simulated hour
        """
        produced_kwh = self.solar_capacity_kw * sun_factor * self.generation_efficiency
        # charge battery with surplus
        self.battery_soc_kwh = min(self.battery_capacity_kwh, self.battery_soc_kwh + produced_kwh)
        self.load_log.append({"type": "solar_prod", "kwh": produced_kwh, "time": time.time()})
        return produced_kwh

    def consume(self, kwh: float) -> bool:
        """
        Attempt to consume kwh from battery. Returns True if supply available, else False.
        """
        if kwh <= self.battery_soc_kwh + 1e-9:
            self.battery_soc_kwh -= kwh
            self.load_log.append({"type": "consumed", "kwh": kwh, "time": time.time()})
            return True
        else:
            # insufficient energy
            self.load_log.append({"type": "failed_consume", "kwh": kwh, "time": time.time()})
            return False

    def status(self) -> Dict:
        return {
            "battery_soc_kwh": round(self.battery_soc_kwh, 3),
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "solar_capacity_kw": self.solar_capacity_kw,
        }

# ------------------------------
# Ion-exchange unit (perchlorates removal)
# ------------------------------
@dataclass
class IonExchangeUnit:
    resin_capacity_per_cycle: float = 1.0  # how many liters (or parcels) processed before regeneration (abstract)
    current_load: float = 0.0
    energy_per_regen_kwh: float = 0.5
    brine_byproduct_l: float = 0.0

    async def process_parcel(self, parcel: Dict, energy_system: EnergySystem) -> Optional[Dict]:
        """
        parcel: {'id':..., 'contaminants': [...], 'volume_l':...}
        returns filtered parcel or None if fails
        """
        # simulate processing energy cost small
        process_energy = 0.02
        ok = energy_system.consume(process_energy)
        if not ok:
            print("[IonExchange] Not enough energy to run processing step.")
            return None
        self.current_load += 1.0
        await asyncio.sleep(0.3)
        # simulate perchlorate removal success (high)
        cont = [c for c in parcel.get("contaminants", []) if c != "perchlorate"]
        parcel_filtered = dict(parcel)
        parcel_filtered["contaminants"] = cont
        # if resin saturated, need regen
        if self.current_load >= self.resin_capacity_per_cycle:
            # require regeneration (consumes energy and produces brine)
            regen_ok = energy_system.consume(self.energy_per_regen_kwh)
            if regen_ok:
                self.brine_byproduct_l += 0.5  # abstract amount
                self.current_load = 0.0
                print("[IonExchange] Resin regenerated; brine produced.")
            else:
                print("[IonExchange] Not enough energy to regenerate resin. Unit offline.")
                return None
        return parcel_filtered

# ------------------------------
# Optional Deep Learning Classifier Hook
# ------------------------------
class DLClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.backend = DL_BACKEND
        self.model_path = model_path
        self.model = None
        if self.backend and model_path and os.path.exists(model_path):
            try:
                if self.backend == "tensorflow":
                    self.model = tf.keras.models.load_model(model_path)
                elif self.backend == "torch":
                    self.model = torch.load(model_path, map_location='cpu')
                print(f"[DLClassifier] Loaded model from {model_path} using {self.backend}")
            except Exception as e:
                print("[DLClassifier] Failed to load model:", e)
                self.model = None
        else:
            if self.backend:
                print("[DLClassifier] No model file provided or not found; running in fallback (rules).")
            else:
                print("[DLClassifier] No DL backend available; running in fallback (rules).")

    def classify(self, item: WasteItem) -> str:
        # If model available, you'd run inference on item.image_path here.
        if self.model and item.image_path:
            try:
                # placeholder: real inference code depends on model format
                # e.g., tf: preprocess image, expand dims, predict -> map to class
                pass
            except Exception as e:
                print("[DLClassifier] inference error:", e)
        # fallback: use rule-based classification using textual label
        name = item.raw_type.lower()
        if any(k in name for k in ["plastic", "bottle", "wrapper", "pouch", "ldpe", "hdpe", "pet"]):
            return "plastic"
        if any(k in name for k in ["steel", "iron", "nut", "bolt", "screw"]):
            return "ferrous_metal"
        if any(k in name for k in ["aluminum", "aluminium", "can", "foil"]):
            return "nonferrous_metal"
        if any(k in name for k in ["cloth", "shirt", "fabric", "towel", "eva"]):
            return "fabric"
        if any(k in name for k in ["foam", "epe", "polystyrene", "zotek", "foam"]):
            return "foam"
        if any(k in name for k in ["glass", "jar"]):
            return "glass"
        if any(k in name for k in ["food", "organic", "leftover"]):
            return "organic"
        return random.choice(["plastic", "fabric", "unknown"])

# ------------------------------
# Processing modules (updated to include energy usage)
# ------------------------------
class Conveyor:
    async def feed_items(self, items: List[WasteItem], queue: asyncio.Queue):
        for it in items:
            print(f"[Conveyor] Feeding item #{it.id}: {it.raw_type}")
            await queue.put(it)
            await asyncio.sleep(0.15)
        await queue.put(None)
        print("[Conveyor] All items fed.")

class SmartSorter:
    def __init__(self, classifier: DLClassifier, energy_system: EnergySystem):
        self.classifier = classifier
        self.energy_system = energy_system

    async def sort(self, queue_in: asyncio.Queue, queue_out: Dict[str, asyncio.Queue]):
        while True:
            item = await queue_in.get()
            if item is None:
                for q in queue_out.values():
                    await q.put(None)
                print("[Sorter] No more items. Shutting sorter.")
                break
            # small energy cost per classification
            classify_energy = 0.005
            if not self.energy_system.consume(classify_energy):
                print("[Sorter] Low energy, classification skipped -> manual_check")
                await queue_out["manual_check"].put(item)
                continue
            classification = self.classifier.classify(item)
            if classification == "ferrous_metal":
                target = "magnetic"
            elif classification == "nonferrous_metal":
                target = "eddy"
            elif classification == "plastic":
                target = "air"
            elif classification in ("fabric", "foam"):
                target = "fabric_bin"
            elif classification == "glass":
                target = "glass_bin"
            elif classification == "organic":
                target = "organic_bin"
            else:
                target = "manual_check"
            print(f"[Sorter] Item #{item.id} classified as {classification} -> routing to {target}")
            await queue_out[target].put(item)
            await asyncio.sleep(0.08)

class MagneticSeparator:
    def __init__(self, energy_system: EnergySystem):
        self.energy_system = energy_system

    async def run(self, queue_in: asyncio.Queue, out_bin: List[WasteItem]):
        while True:
            item = await queue_in.get()
            if item is None:
                print("[Magnetic] Done.")
                break
            if not self.energy_system.consume(0.02):
                print("[Magnetic] Not enough energy to operate, skipping item.")
                continue
            await asyncio.sleep(0.2)
            out_bin.append(item)
            print(f"[Magnetic] Collected ferrous #{item.id}")

class EddyCurrentSeparator:
    def __init__(self, energy_system: EnergySystem):
        self.energy_system = energy_system

    async def run(self, queue_in: asyncio.Queue, out_bin: List[WasteItem]):
        while True:
            item = await queue_in.get()
            if item is None:
                print("[Eddy] Done.")
                break
            if not self.energy_system.consume(0.03):
                print("[Eddy] Low energy, skipping.")
                continue
            await asyncio.sleep(0.25)
            out_bin.append(item)
            print(f"[Eddy] Collected nonferrous #{item.id}")

class AirClassifier:
    def __init__(self, energy_system: EnergySystem):
        self.energy_system = energy_system

    async def run(self, queue_in: asyncio.Queue, pellet_queue: asyncio.Queue, recalc_bin: List[WasteItem]):
        while True:
            item = await queue_in.get()
            if item is None:
                await pellet_queue.put(None)
                print("[AirClassifier] Done.")
                break
            if not self.energy_system.consume(0.025):
                print("[AirClassifier] Not enough energy -> manual bin")
                recalc_bin.append(item)
                continue
            await asyncio.sleep(0.35)
            if random.random() < 0.06:
                recalc_bin.append(item)
                print(f"[AirClassifier] Rejected suspicious item #{item.id} -> manual bin")
            else:
                await pellet_queue.put(item)
                print(f"[AirClassifier] Sent plastic #{item.id} to pelletizer")

class Pelletizer:
    def __init__(self, energy_system: EnergySystem):
        self.energy_system = energy_system

    async def run(self, queue_in: asyncio.Queue, pellets_store: List[Dict]):
        while True:
            item = await queue_in.get()
            if item is None:
                print("[Pelletizer] Done.")
                break
            # energy to shred+extrude
            if not self.energy_system.consume(0.12):
                print("[Pelletizer] Not enough energy to pelletize, item queued.")
                await asyncio.sleep(0.5)
                await queue_in.put(item)  # retry later
                continue
            await asyncio.sleep(0.6)
            pellet = {"from": item.id, "mass_kg": item.mass_kg, "type": "pellet"}
            pellets_store.append(pellet)
            print(f"[Pelletizer] Produced pellet from #{item.id}")

class ActivatedCarbonUnit:
    def __init__(self, energy_system: EnergySystem):
        self.energy_system = energy_system

    async def run(self, input_flow: List[Dict], filtered_store: List[Dict]):
        for parcel in input_flow:
            # small energy cost
            if not self.energy_system.consume(0.05):
                print("[ActivatedCarbon] Low energy - skipping parcel for now.")
                continue
            await asyncio.sleep(0.35)
            parcel_filtered = dict(parcel)
            parcel_filtered["filtered_by"] = "activated_carbon"
            filtered_store.append(parcel_filtered)
            print(f"[ActivatedCarbon] Filtered water parcel id={parcel.get('id')}")
        print("[ActivatedCarbon] Complete.")

class ThreeDPrinter:
    def __init__(self, energy_system: EnergySystem):
        self.queue: List[Dict] = []
        self.energy_system = energy_system

    async def submit_print_job(self, item_description: str, pellet_cost: int, pellets_store: List[Dict]):
        # printing energy cost per job ~ 0.3 kWh
        job_energy = 0.3
        if not self.energy_system.consume(job_energy):
            print(f"[3DPrinter] Not enough energy to start print job '{item_description}'.")
            return False
        if len(pellets_store) < pellet_cost:
            print(f"[3DPrinter] Not enough pellets to print {item_description}. Needed {pellet_cost}, have {len(pellets_store)}")
            return False
        consumed = [pellets_store.pop(0) for _ in range(pellet_cost)]
        await asyncio.sleep(0.9)
        print(f"[3DPrinter] Printed '{item_description}' using {len(consumed)} pellets.")
        return True

# ------------------------------
# Monitoring & Orchestration
# ------------------------------
@dataclass
class Metrics:
    energy_kwh: float = 0.0
    processed_count: int = 0

    def log_energy(self, v: float):
        self.energy_kwh += v

# ------------------------------
# Orchestration main (async)
# ------------------------------
async def run_simulation(sample_items: List[WasteItem],
                         energy_sys: EnergySystem,
                         classifier: DLClassifier,
                         run_hours: int = 6):
    """
    Runs an async simulation:
    - runs conveyor + sorter + separators + pelletizer
    - simulates solar production per hour (sun_factor random to simulate dust)
    - processes water parcels through activated carbon then ion-exchange
    """
    # queues and stores
    sorter_queue = asyncio.Queue()
    queues = {
        "magnetic": asyncio.Queue(),
        "eddy": asyncio.Queue(),
        "air": asyncio.Queue(),
        "fabric_bin": asyncio.Queue(),
        "glass_bin": asyncio.Queue(),
        "organic_bin": asyncio.Queue(),
        "manual_check": asyncio.Queue(),
    }
    pellet_queue = asyncio.Queue()
    pellets_store: List[Dict] = []
    ferrous_bin: List[WasteItem] = []
    nonferrous_bin: List[WasteItem] = []
    recalc_bin: List[WasteItem] = []
    fabric_bin_store: List[WasteItem] = []
    glass_bin_store: List[WasteItem] = []
    organic_bin_store: List[WasteItem] = []
    water_input_parcels: List[Dict] = [{"id": f"w{i+1}", "contaminants": ["organic", "dust", "perchlorate"], "volume_l": 1.0} for i in range(2)]
    filtered_water = []

    # instantiate modules
    conveyor = Conveyor()
    sorter = SmartSorter(classifier, energy_sys)
    mag = MagneticSeparator(energy_sys)
    eddy = EddyCurrentSeparator(energy_sys)
    air = AirClassifier(energy_sys)
    pelletizer = Pelletizer(energy_sys)
    activated = ActivatedCarbonUnit(energy_sys)
    ion_ex = IonExchangeUnit(resin_capacity_per_cycle=2.0, energy_per_regen_kwh=0.4)
    printer = ThreeDPrinter(energy_sys)

    # tasks
    tasks = [
        asyncio.create_task(conveyor.feed_items(sample_items, sorter_queue)),
        asyncio.create_task(sorter.sort(sorter_queue, queues)),
        asyncio.create_task(mag.run(queues["magnetic"], ferrous_bin)),
        asyncio.create_task(eddy.run(queues["eddy"], nonferrous_bin)),
        asyncio.create_task(air.run(queues["air"], pellet_queue, recalc_bin)),
        asyncio.create_task(pelletizer.run(pellet_queue, pellets_store)),
        asyncio.create_task(drain_bin(queues["fabric_bin"], fabric_bin_store, label="FabricBin")),
        asyncio.create_task(drain_bin(queues["glass_bin"], glass_bin_store, label="GlassBin")),
        asyncio.create_task(drain_bin(queues["organic_bin"], organic_bin_store, label="OrganicBin")),
    ]

    # simulate solar power over run_hours (each loop = 1 hour)
    for hour in range(run_hours):
        # realistic sun_factor: between 0.2 (dusty morning) and 1.0 (clear midday)
        sun_factor = max(0.05, random.gauss(0.7, 0.2))
        produced = energy_sys.produce_solar(sun_factor)
        print(f"[Solar] Hour {hour+1}: produced {produced:.3f} kWh (sun_factor={sun_factor:.2f}). Battery SOC: {energy_sys.battery_soc_kwh:.3f} kWh")
        await asyncio.sleep(0.05)  # let simulation breathe
        # if battery low, skip some heavy tasks or attempt regeneration later (handled in modules)

    # wait for processing tasks to finish
    await asyncio.gather(*tasks)

    # activated carbon + ion exchange chain for water parcels
    await activated.run(water_input_parcels, filtered_water)
    # pass through ion-exchange (removes perchlorate)
    final_water = []
    for parcel in filtered_water:
        res = await ion_ex.process_parcel(parcel, energy_sys)
        if res:
            final_water.append(res)
    print(f"[Water] Final water parcels after ion-exchange: {[p['id'] for p in final_water]}")

    # simulate some print jobs
    await printer.submit_print_job("Repair Wrench", pellet_cost=2, pellets_store=pellets_store)
    await printer.submit_print_job("Small Storage Box", pellet_cost=3, pellets_store=pellets_store)

    # final report
    report = {
        "ferrous": [i.id for i in ferrous_bin],
        "nonferrous": [i.id for i in nonferrous_bin],
        "fabric": [i.id for i in fabric_bin_store],
        "glass": [i.id for i in glass_bin_store],
        "organic": [i.id for i in organic_bin_store],
        "recalc": [i.id for i in recalc_bin],
        "pellets": len(pellets_store),
        "final_water": [p["id"] for p in final_water],
        "battery": energy_sys.status(),
        "ion_brine_l": ion_ex.brine_byproduct_l
    }
    return report

# helper drain
async def drain_bin(q: asyncio.Queue, store: List[WasteItem], label="Bin"):
    while True:
        item = await q.get()
        if item is None:
            print(f"[{label}] Done.")
            break
        await asyncio.sleep(0.12)
        store.append(item)
        print(f"[{label}] Stored item #{item.id}")

# ------------------------------
# Streamlit app wrapper
# ------------------------------
def build_sample_items_from_textlist(texts: List[str]) -> List[WasteItem]:
    items = []
    for i, t in enumerate(texts):
        items.append(WasteItem(id=i+1, name=t, raw_type=t, mass_kg=0.1))
    return items

def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Install with: pip install streamlit")
        return

    st.set_page_config(page_title="MARS-CYCLE Demo", layout="wide")
    st.title("MARS-CYCLE: Smart Recycling Machine (Simulation)")

    st.sidebar.header("Simulation parameters")
    solar_kw = st.sidebar.number_input("Solar peak capacity (kW)", value=2.0, step=0.5)
    battery_kwh = st.sidebar.number_input("Battery capacity (kWh)", value=10.0, step=1.0)
    run_hours = st.sidebar.slider("Simulated daylight hours (iterations)", min_value=1, max_value=24, value=6)

    # sample items input
    st.sidebar.header("Input items (comma-separated labels)")
    default_items = "Plastic Bottle, Aluminum Can, Shirt, Zotek Foam, Food Scraps, Steel Bolt, Glass Jar"
    txt = st.sidebar.text_area("Items", value=default_items, height=120)
    item_texts = [s.strip() for s in txt.split(",") if s.strip()]
    sample_items = build_sample_items_from_textlist(item_texts)

    # model path (optional)
    st.sidebar.header("DL Classifier (optional)")
    model_path = st.sidebar.text_input("Model path (optional)", value="")

    if st.sidebar.button("Run simulation"):
        # create energy system & classifier
        energy_sys = EnergySystem(solar_capacity_kw=solar_kw, battery_capacity_kwh=battery_kwh,
                                  battery_soc_kwh=battery_kwh)
        classifier = DLClassifier(model_path if model_path else None)
        # run async loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        report = loop.run_until_complete(run_simulation(sample_items, energy_sys, classifier, run_hours=run_hours))
        loop.close()

        st.subheader("Simulation Report")
        st.write(report)
        st.metric("Battery SOC (kWh)", report["battery"]["battery_soc_kwh"])
        st.metric("Pellets produced", report["pellets"])
        st.metric("Final Water Parcels", len(report["final_water"]))
        st.write("Ion-exchange brine produced (L):", report["ion_brine_l"])

    st.info("Notes: This is a proof-of-concept demo. For real DL classification, provide a trained model path and images.")

# ------------------------------
# CLI entry
# ------------------------------
def cli_demo():
    # sample items
    sample_items = [
        WasteItem(1, "Plastic Bottle", "plastic bottle", 0.05),
        WasteItem(2, "Aluminum Can", "aluminum can", 0.02),
        WasteItem(3, "Used Shirt", "old shirt", 0.25),
        WasteItem(4, "EVA Foam Pack", "zotek foam", 0.4),
        WasteItem(5, "Food Leftover", "food scrap", 0.15),
        WasteItem(6, "Steel Bolt", "steel bolt", 0.03),
        WasteItem(7, "Glass Jar", "glass jar", 0.2),
    ]
    energy_sys = EnergySystem(solar_capacity_kw=2.0, battery_capacity_kwh=10.0, battery_soc_kwh=10.0)
    classifier = DLClassifier(None)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    report = loop.run_until_complete(run_simulation(sample_items, energy_sys, classifier, run_hours=6))
    loop.close()
    print("\n--- CLI Report ---")
    print(report)

if __name__ == "__main__":
    # if streamlit is available and user runs with `streamlit run script.py`, Streamlit will call run_streamlit_app
    # but also allow CLI execution for quick demo.
    if "streamlit" in sys.argv[0]:
        # when run via `streamlit run`, call the app builder
        run_streamlit_app()
    else:
        print("Run as CLI demo or with Streamlit.")
        print("To run CLI demo: python mars_recycling_sim_full.py")
        print("To run Streamlit UI (recommended for demo): streamlit run mars_recycling_sim_full.py")

        # default to CLI demo
        cli_demo()
