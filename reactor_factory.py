# SPDX-FileCopyrightText: 2024-present The Bluemira Team
#
# SPDX-License-Identifier: MIT
# %%
import json
import time
from pathlib import Path
from typing import Self

from bluemira.base.components import Component
from bluemira.base.designer import run_designer
from bluemira.base.file import get_bluemira_root
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.reactor_config import ReactorConfig
from bluemira.builders.plasma import Plasma, PlasmaBuilder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.flux_surfaces import ClosedFluxSurface
from bluemira.geometry.tools import (
    interpolate_bspline,
)
from bluemira.materials.cache import establish_material_cache
from bluemira.radiation_transport.neutronics.zero_d_neutronics import (
    ZeroDNeutronicsModel,
)
from eudemo.blanket import Blanket
from eudemo.comp_managers import (
    CoilStructures,
    Cryostat,
    CryostatThermalShield,
    RadiationShield,
    ThermalShield,
    VacuumVesselThermalShield,
)
from eudemo.equilibria import (
    DummyFixedEquilibriumDesigner,
    FixedEquilibriumDesigner,
)
from eudemo.equilibria.stability import run_vertical_stability_calculation
from eudemo.ivc import IVCShapes, design_ivc
from eudemo.maintenance.equatorial_port import EquatorialPortKOZDesigner
from eudemo.maintenance.lower_port import LowerPortKOZDesigner
from eudemo.maintenance.upper_port import UpperPortKOZDesigner
from eudemo.model_managers import EquilibriumManager, NeutronicsManager
from eudemo.neutronics.run import run_csg_neutronics, run_dagmc_neutronics
from eudemo.params import EUDEMOReactorParams
from eudemo.pf_coils import PFCoil as EU_PFCoil
from eudemo.power_cycle import SteadyStatePowerCycleSolver
from eudemo.radial_build import radial_build as eudemo_radial_build
from eudemo.reactor import EUDEMO
from eudemo.tf_coils import TFCoil as EU_TFCoil
from matproplib.conditions import OperationalConditions

from bluemira_st.blanket.manager import BB
from bluemira_st.build_routines import SphericalReactor
from bluemira_st.params import BluemiraSTParams
from bluemira_st.pf_coil.manager import PFCoil as SPH_PFCoil
from bluemira_st.radial_build.run_process import radial_build as st_radial_build
from bluemira_st.tf_coil.manager import TFCoil as SPH_TFCoil


class FactoryConfig:
    """Class to load ReactorFactory config file."""

    @staticmethod
    def load_config(build_config: str | Path | dict) -> ReactorConfig:

        with Path(build_config).open("r", encoding="utf-8") as f:
            data = json.load(f)

        reactor_type: str = data["machine_type"]
        match reactor_type:
            case "eudemo":
                return reactor_type, ReactorConfig(build_config, EUDEMOReactorParams)
            case "spherical_reactor":
                return reactor_type, ReactorConfig(build_config, BluemiraSTParams)


class ReactorFactory:
    """A reactor creation wrapper with two components."""

    reactor_type: str
    reactor_config: ReactorConfig

    def __init__(self, reactor_type: str, reactor_config: ReactorConfig) -> None:
        self.reactor_type = reactor_type
        self.reactor_config = reactor_config

    @classmethod
    def from_config_file(cls, build_config: str | Path | dict) -> Self:
        reactor_type, reactor_config = FactoryConfig.load_config(build_config)
        return cls(reactor_type, reactor_config)

    def create_reactor(self) -> SphericalReactor | EUDEMO:
        reactor: SphericalReactor | EUDEMO
        match self.reactor_type:
            case "eudemo":
                reactor = ReactorFactory._create_eudemo(self.reactor_config)
            case "spherical_reactor":
                reactor = ReactorFactory._create_spherical_reactor(self.reactor_config)
        return reactor

    @staticmethod
    def _build_reference_equilibrium(
        reactor: SphericalReactor | EUDEMO,
        reactor_config: ReactorConfig,
        lcfs_coords: list | None = None,
        profiles: list | None = None,
    ) -> Equilibrium:
        params_obj = reactor_config.params_for("free_boundary_equilibrium")
        config = reactor_config.config_for("free_boundary_equilibrium")

        if lcfs_coords is not None and profiles is not None:
            reactor.equilibria = EquilibriumManager()
            return EUDEMO.build_reference_equilibrium(
                params_obj,
                config,
                reactor.equilibria,
                lcfs_coords,
                profiles,
            )
        return SphericalReactor.build_reference_equilibrium(
            params_obj.global_params, config
        )

    @staticmethod
    def _build_plasma(
        reactor_config: ReactorConfig,
        reference_eq: Equilibrium,
    ) -> Plasma:
        params = reactor_config.params_for("free_boundary_equilibrium")
        config = reactor_config.config_for("free_boundary_equilibrium")
        lcfs_loop = reference_eq.get_LCFS()
        lcfs_wire = interpolate_bspline(
            {"x": lcfs_loop.x, "z": lcfs_loop.z}, closed=True
        )
        builder = PlasmaBuilder(params, config, lcfs_wire)
        return Plasma(builder.build())

    @staticmethod
    def _build_pf_coils(
        reactor: SphericalReactor | EUDEMO,
        reactor_config: ReactorConfig,
        reference_eq: Equilibrium | None = None,
        pf_coil_keep_out_zones: list | None = None,
    ) -> EU_PFCoil | SPH_PFCoil:
        params = reactor_config.params_for("pf_coils")
        config = reactor_config.config_for("pf_coils")
        if pf_coil_keep_out_zones is not None:
            return EUDEMO.build_pf_coils(
                params,
                config,
                reactor.equilibria,
                reactor.tf_coils.xz_outer_boundary,
                pf_coil_keep_out_zones,
            )
        return SphericalReactor.build_pf_coils(
            params,
            config,
            reference_eq.coilset,
        )

    @staticmethod
    def _build_tf_coils(
        reactor: SphericalReactor | EUDEMO,
        reactor_config: ReactorConfig,
        reference_eq: Equilibrium | None = None,
        vv_thermal_shield: VacuumVesselThermalShield | None = None,
    ) -> EU_TFCoil | SPH_TFCoil:
        params = reactor_config.params_for("tf_coils")
        config = reactor_config.config_for("tf_coils")
        if vv_thermal_shield is not None:
            return EUDEMO.build_tf_coils(
                params,
                config,
                reactor.plasma.lcfs(),
                vv_thermal_shield.xz_boundary,
            )
        return SphericalReactor.build_tf_coils(
            params,
            config,
            reference_eq.coilset,
            interpolate_bspline(reference_eq.get_LCFS(), closed=True),
        )

    @staticmethod
    def _build_blankets(
        reactor_config: ReactorConfig,
        reference_eq: Equilibrium | None = None,
        ivc_shapes: IVCShapes | None = None,
        r_inner_cut: float | None = None,
        cut_angle: float | None = None,
    ) -> BB | Blanket:
        params = reactor_config.params_for("blanket")
        config = reactor_config.config_for("blanket")

        if ivc_shapes is not None and r_inner_cut is not None and cut_angle is not None:
            return EUDEMO.build_blanket(
                params,
                config,
                ivc_shapes.inner_boundary,
                ivc_shapes.blanket_face,
                r_inner_cut,
                cut_angle,
            )
        return SphericalReactor.build_bb(
            params,
            config,
            mat_name="BB_BZ_MATERIAL",
            ref_fbe=reference_eq,
        )

    @staticmethod
    def _build_vacuum_vessel_thermal_shield(
        reactor_config: ReactorConfig, vacuum_vessel
    ) -> VacuumVesselThermalShield:
        return EUDEMO.build_vacuum_vessel_thermal_shield(
            reactor_config.params_for("thermal_shield"),
            reactor_config.config_for("thermal_shield", "VVTS"),
            vacuum_vessel.xz_boundary,
        )

    @staticmethod
    def _design_ivc(
        reactor_config: ReactorConfig, reference_eq: Equilibrium
    ) -> IVCShapes:
        return design_ivc(
            reactor_config.params_for("IVC").global_params,
            reactor_config.config_for("IVC"),
            equilibrium=reference_eq,
        )

    @staticmethod
    def _build_vacuum_vessel(reactor_config: ReactorConfig, ivc_shapes: IVCShapes):
        return EUDEMO.build_vacuum_vessel(
            reactor_config.params_for("vacuum_vessel"),
            reactor_config.config_for("vacuum_vessel"),
            ivc_shapes.outer_boundary,
        )

    @staticmethod
    def _build_divertor(reactor_config: ReactorConfig, ivc_shapes: IVCShapes):
        return EUDEMO.build_divertor(
            reactor_config.params_for("divertor"),
            reactor_config.config_for("divertor"),
            ivc_shapes.divertor_face,
        )

    @staticmethod
    def _upper_port_designer(reactor_config: ReactorConfig, ivc_shapes: IVCShapes):
        designer = UpperPortKOZDesigner(
            reactor_config.params_for("upper_port"),
            reactor_config.config_for("upper_port"),
            ivc_shapes.blanket_face,
        )
        return designer.execute()

    @staticmethod
    def _build_cryots(
        reactor_config: ReactorConfig, pf_coils: EU_PFCoil, tf_coils: EU_TFCoil
    ) -> CryostatThermalShield:
        return EUDEMO.build_cryots(
            reactor_config.params_for("thermal_shield"),
            reactor_config.config_for("thermal_shield", "cryostat"),
            pf_coils.xz_boundary,
            tf_coils.xz_outer_boundary,
        )

    @staticmethod
    def _assemble_thermal_shield(
        vv_thermal_shield: VacuumVesselThermalShield,
        cryostat_thermal_shield: CryostatThermalShield,
    ) -> ThermalShield:
        return EUDEMO.assemble_thermal_shield(vv_thermal_shield, cryostat_thermal_shield)

    @staticmethod
    def _build_coil_structures(
        reactor_config: ReactorConfig,
        pf_coils: EU_PFCoil,
        tf_coils: EU_TFCoil,
        keep_out_zones: list,
    ) -> CoilStructures:
        return EUDEMO.build_coil_structures(
            reactor_config.params_for("coil_structures"),
            reactor_config.config_for("coil_structures"),
            tf_coil_xz_face=tf_coils.xz_face,
            pf_coil_xz_wires=pf_coils.PF_xz_boundary,
            pf_coil_keep_out_zones=keep_out_zones,
        )

    @staticmethod
    def _build_cryostat(
        reactor_config: ReactorConfig, cryostat_thermal_shield: CryostatThermalShield
    ) -> Cryostat:
        return EUDEMO.build_cryostat(
            reactor_config.params_for("cryostat"),
            reactor_config.config_for("cryostat"),
            cryostat_thermal_shield.xz_boundary,
        )

    @staticmethod
    def _build_radiation_shield(reactor_config: ReactorConfig, cryostat):
        return EUDEMO.build_radiation_shield(
            reactor_config.params_for("radiation_shield"),
            reactor_config.config_for("radiation_shield"),
            cryostat.xz_boundary,
        )

    @staticmethod
    def _build_upper_port(
        reactor_config: ReactorConfig,
        upper_port_koz_xz,
        pf_coils: EU_PFCoil,
        cryostat_thermal_shield: CryostatThermalShield,
    ):
        return EUDEMO.build_upper_port(
            reactor_config.params_for("upper_port"),
            reactor_config.config_for("upper_port"),
            upper_port_koz_xz,
            pf_coils,
            cryostat_thermal_shield.xz_boundary,
        )

    @staticmethod
    def _build_equatorial_port(
        reactor_config: ReactorConfig, cryostat_thermal_shield: CryostatThermalShield
    ):
        return EUDEMO.build_equatorial_port(
            reactor_config.params_for("equatorial_port"),
            reactor_config.config_for("equatorial_port"),
            cryostat_thermal_shield.xz_boundary,
        )

    @staticmethod
    def _build_lower_port(
        reactor_config: ReactorConfig,
        lp_duct_angled_nowall_extrude_boundary,
        lp_duct_straight_nowall_extrude_boundary,
        cryostat,
    ):
        return EUDEMO.build_lower_port(
            reactor_config.params_for("lower_port"),
            reactor_config.config_for("lower_port"),
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
            cryostat.xz_boundary,
        )

    @staticmethod
    def _build_cryostat_plugs(
        reactor_config: ReactorConfig, ports: list, cryostat: Cryostat
    ) -> Component:
        return EUDEMO.build_cryostat_plugs(
            reactor_config.params_for("cryostat"),
            reactor_config.config_for("cryostat"),
            ports,
            cryostat.xz_boundary,
        )

    @staticmethod
    def _build_radiation_plugs(
        reactor_config: ReactorConfig,
        cr_plugs: Component,
        radiation_shield: RadiationShield,
    ) -> Component:
        return EUDEMO.build_radiation_plugs(
            reactor_config.params_for("radiation_shield"),
            reactor_config.config_for("radiation_shield"),
            cr_plugs,
            radiation_shield.xz_boundary,
        )

    @staticmethod
    def _run_dagmc_neutronics(
        reactor: EUDEMO, reactor_config: ReactorConfig, reference_eq: Equilibrium
    ):
        return run_dagmc_neutronics(
            reactor,
            reactor_config.params_for("neutronics", "DAGMC").global_params,
            reactor_config.config_for("neutronics", "DAGMC"),
            reference_eq,
        )

    @staticmethod
    def _eq_port_designer(reactor_config: ReactorConfig, x_ob: float) -> list:
        eq_port_designer = EquatorialPortKOZDesigner(
            reactor_config.params_for("equatorial_port"),
            reactor_config.config_for("equatorial_port"),
            x_ob,
        )
        return eq_port_designer.execute()

    @staticmethod
    def _lower_port_designer(
        reactor_config: ReactorConfig, ivc_shapes, tf_coils
    ) -> list:
        return LowerPortKOZDesigner(
            reactor_config.params_for("lower_port"),
            reactor_config.config_for("lower_port"),
            ivc_shapes.divertor_face,
            ivc_shapes.div_wall_join_pt,
            tf_coils.xz_outer_boundary,
        ).execute()

    @staticmethod
    def calculate_centre_of_mass() -> None:
        return

    @staticmethod
    def _check_param_exists(reactor_config: ReactorConfig, param: str) -> bool:
        reactor_config.params_for(param)
        return False

    @staticmethod
    def _create_spherical_reactor(reactor_config: ReactorConfig) -> SphericalReactor:
        """Reactor function."""
        reactor = SphericalReactor(
            "Bluemira Spherical Tokamak Example",
            n_sectors=reactor_config.global_params.n_TF.value,
        )

        establish_material_cache([
            "bluemira_st.materials",
            "matproplib",
            Path(get_bluemira_root(), "examples", "design", "design_materials.py")
            .resolve()
            .as_posix(),
        ])

        params = reactor_config.params_for("radial_build")
        config = reactor_config.config_for("radial_build")
        st_radial_build(params.global_params, config)
        reference_eq = ReactorFactory._build_reference_equilibrium(
            reactor, reactor_config
        )

        reactor.plasma = ReactorFactory._build_plasma(reactor_config, reference_eq)
        ReactorFactory._check_param_exists(reactor_config, "ch")
        reactor.pf_coils = ReactorFactory._build_pf_coils(reactor, reactor_config)
        reactor.tf_coils = ReactorFactory._build_tf_coils(reactor, reactor_config)
        reactor.blanket = ReactorFactory._build_blankets(reactor_config)

        reactor.inboard_shield = SphericalReactor.build_is(
            reactor_config.params_for("inboard_shield"),
            reactor_config.config_for("inboard_shield"),
            mat_name="EUROFER_MAT",
            ref_fbe=reference_eq,
        )
        # reactor.show_cad("xyz")
        # reactor.show_cad("xz")

        return reactor

    @staticmethod
    def _create_eudemo(reactor_config: ReactorConfig) -> EUDEMO:
        run_time_track = {
            "Total": 0.0,
            "PROCESS": 0.0,
            "CSG neutronics": 0.0,
            "CAD neutronics": 0.0,
        }
        reactor = EUDEMO("EUDEMO", n_sectors=reactor_config.global_params.n_TF.value)

        start = time.time()
        establish_material_cache([
            "eudemo.materials",
            "eurofusion_materials.library",
            "matproplib",
        ])
        params = reactor_config.params_for("radial_build")
        config = reactor_config.config_for("radial_build")
        eudemo_radial_build(params.global_params, config)

        lcfs_coords, profiles = run_designer(
            FixedEquilibriumDesigner,
            reactor_config.params_for("fixed_boundary_equilibrium"),
            reactor_config.config_for("fixed_boundary_equilibrium"),
        )

        lcfs_coords, profiles = run_designer(
            DummyFixedEquilibriumDesigner,
            reactor_config.params_for("dummy_fixed_boundary_equilibrium"),
            reactor_config.config_for("dummy_fixed_boundary_equilibrium"),
        )

        reference_eq = ReactorFactory._build_reference_equilibrium(
            reactor,
            reactor_config,
            lcfs_coords,
            profiles,
        )

        reactor.plasma = ReactorFactory._build_plasma(reactor_config, reference_eq)

        ivc_shapes = ReactorFactory._design_ivc(reactor_config, reference_eq)

        reactor.vacuum_vessel = ReactorFactory._build_vacuum_vessel(
            reactor_config, ivc_shapes
        )

        reactor.divertor = ReactorFactory._build_divertor(reactor_config, ivc_shapes)

        upper_port_koz_xz, r_inner_cut, cut_angle = ReactorFactory._upper_port_designer(
            reactor_config, ivc_shapes
        )

        reactor.blanket = ReactorFactory._build_blankets(
            reactor_config,
            reference_eq,
            ivc_shapes,
            r_inner_cut,
            cut_angle,
        )

        zero_d_params = ZeroDNeutronicsModel(reactor_config.global_params).run()

        reactor_config.global_params.update_from_frame(zero_d_params)

        if reactor_config.config_for("neutronics", "CSG").get("enabled", False):
            neutronics_csg = run_csg_neutronics(
                reactor_config.params_for("neutronics", "CSG").global_params,
                reactor_config.config_for("neutronics", "CSG"),
                blanket=reactor.blanket,
                vacuum_vessel=reactor.vacuum_vessel,
                ivc_shapes=ivc_shapes,
                eq=reference_eq,
                op_cond=OperationalConditions(temperature=298, pressure=101325),
            )
            if reactor_config.config_for("neutronics", "CSG")["show_data"]:
                reactor.neutronics.plot()
                bluemira_print(f"{reactor.neutronics}")
        else:
            neutronics_csg = None

        reactor.neutronics = NeutronicsManager(zero_d_params, neutronics_csg)

        vv_thermal_shield = ReactorFactory._build_vacuum_vessel_thermal_shield(
            reactor_config,
            reactor.vacuum_vessel,
        )

        reactor.tf_coils, peak_opt_ripple = ReactorFactory._build_tf_coils(
            reactor,
            reactor_config,
            reactor.plasma.lcfs(),
            vv_thermal_shield,
        )
        reactor_config.global_params.TF_peak_ripple_opt.set_value(
            peak_opt_ripple, "BLUEMIRA"
        )

        eq_port_koz_xz = ReactorFactory._eq_port_designer(reactor_config, x_ob=20.0)

        (
            _lp_duct_xz_void_space,
            lower_port_koz_xz,
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
        ) = ReactorFactory._lower_port_designer(
            reactor_config, ivc_shapes, reactor.tf_coils
        )

        reactor.pf_coils = ReactorFactory._build_pf_coils(
            reactor,
            reactor_config,
            reference_eq,
            [
                upper_port_koz_xz,
                eq_port_koz_xz,
                lower_port_koz_xz,
            ],
        )
        run_vertical_stability_calculation(
            reactor_config.params_for("vertical_stability").global_params,
            reactor_config.config_for("vertical_stability"),
            reactor.equilibria.get_state(reactor.equilibria.SOF).eq,
            reactor.vacuum_vessel.xz_boundary,
            reactor.vacuum_vessel.xz_inner_boundary,
            [upper_port_koz_xz, eq_port_koz_xz, lower_port_koz_xz],
        )

        cryostat_thermal_shield = ReactorFactory._build_cryots(
            reactor_config,
            reactor.pf_coils,
            reactor.tf_coils,
        )

        reactor.thermal_shield = ReactorFactory._assemble_thermal_shield(
            vv_thermal_shield, cryostat_thermal_shield
        )

        reactor.coil_structures = ReactorFactory._build_coil_structures(
            reactor_config,
            tf_coils=reactor.tf_coils,
            pf_coils=reactor.pf_coils,
            keep_out_zones=[
                upper_port_koz_xz,
                eq_port_koz_xz,
                lower_port_koz_xz,
            ],
        )

        reactor.cryostat = ReactorFactory._build_cryostat(
            reactor_config,
            cryostat_thermal_shield,
        )

        reactor.radiation_shield = ReactorFactory._build_radiation_shield(
            reactor_config,
            reactor.cryostat,
        )

        ts_upper_port, vv_upper_port = ReactorFactory._build_upper_port(
            reactor_config,
            upper_port_koz_xz,
            reactor.pf_coils,
            cryostat_thermal_shield,
        )
        ts_eq_port, vv_eq_port = ReactorFactory._build_equatorial_port(
            reactor_config,
            cryostat_thermal_shield,
        )

        ts_lower_port, vv_lower_port = ReactorFactory._build_lower_port(
            reactor_config,
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
            reactor.cryostat,
        )

        reactor.vacuum_vessel.add_ports(
            [vv_upper_port, vv_eq_port, vv_lower_port],
            n_TF=reactor_config.global_params.n_TF.value,
        )

        reactor.thermal_shield.add_ports(
            [ts_upper_port, ts_eq_port, ts_lower_port],
            n_TF=reactor_config.global_params.n_TF.value,
        )

        cr_plugs = ReactorFactory._build_cryostat_plugs(
            reactor_config,
            [ts_upper_port, ts_eq_port, ts_lower_port],
            reactor.cryostat,
        )

        rs_plugs = ReactorFactory._build_radiation_plugs(
            reactor_config,
            cr_plugs,
            reactor.radiation_shield,
        )

        reactor.cryostat.add_plugs(
            cr_plugs, n_TF=reactor_config.global_params.n_TF.value
        )

        reactor.radiation_shield.add_plugs(
            rs_plugs, n_TF=reactor_config.global_params.n_TF.value
        )

        reactor.neutronics.dagmc = ReactorFactory._run_dagmc_neutronics(
            reactor,
            reactor_config,
            reference_eq,
        )

        sspc_solver = SteadyStatePowerCycleSolver(reactor_config.global_params)
        sspc_result = sspc_solver.execute()
        reactor_config.global_params.P_el_net.set_value(
            sspc_result["P_el_net"], "BLUEMIRA"
        )

        lcfs = ClosedFluxSurface(reference_eq.get_LCFS())

        reactor_config.global_params.V_p.set_value(lcfs.volume, "BLUEMIRA")

        end = time.time()
        run_time_track["Total"] = end - start
        n_config = reactor_config.config_for("neutronics")
        particles = n_config.get("particles", n_config["DAGMC"]["particles"])
        neutrons = f"{particles:.2g}".replace(".", "_").replace("+", "")
        a_string = f"{reactor_config.global_params.A.value:.3f}".replace(".", "_")
        folder_name = f"results_v05/A_{a_string}_neut_{neutrons}"
        Path(folder_name).mkdir(exist_ok=True, parents=True)
        filename = f"{folder_name}/run_time.json"
        with Path(filename).open("w", encoding="utf-8") as f:
            json.dump(run_time_track, f, indent=2)
        reactor.save_reactor(reactor, reactor_config, folder_name=folder_name)


if __name__ == "__main__":
    """ BUILD_CONFIG_FILE_PATH = Path(
        Path(__file__).parent, "studies/first/config/config.json"
    ).resolve()
    """
    BUILD_CONFIG_FILE_PATH = Path(
        Path(__file__).parent.parent, "bluemira/eudemo/config/build_config.json"
    ).resolve()
    rf = ReactorFactory.from_config_file(BUILD_CONFIG_FILE_PATH)

    reactor = rf.create_reactor()
