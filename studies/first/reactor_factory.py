# SPDX-FileCopyrightText: 2024-present The Bluemira Team
#
# SPDX-License-Identifier: MIT
# %%
from pathlib import Path

from bluemira.base.designer import run_designer
from bluemira.base.file import get_bluemira_root
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.reactor_config import ReactorConfig
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.flux_surfaces import ClosedFluxSurface
from bluemira.geometry.tools import (
    interpolate_bspline,
)
from bluemira.materials.cache import establish_material_cache
from bluemira.radiation_transport.neutronics.zero_d_neutronics import (
    ZeroDNeutronicsModel,
)
from eudemo.equilibria import (
    DummyFixedEquilibriumDesigner,
    FixedEquilibriumDesigner,
)
from eudemo.ivc import design_ivc
from eudemo.maintenance.equatorial_port import EquatorialPortKOZDesigner
from eudemo.maintenance.lower_port import LowerPortKOZDesigner
from eudemo.maintenance.upper_port import UpperPortKOZDesigner
from eudemo.model_managers import EquilibriumManager, NeutronicsManager
from eudemo.neutronics.run import run_csg_neutronics, run_dagmc_neutronics
from eudemo.params import EUDEMOReactorParams
from eudemo.power_cycle import SteadyStatePowerCycleSolver
from eudemo.reactor import (
    EUDEMO,
    assemble_thermal_shield,
    build_blanket,
    build_coil_structures,
    build_cryostat,
    build_cryostat_plugs,
    build_cryots,
    build_divertor,
    build_equatorial_port,
    build_lower_port,
    build_radiation_plugs,
    build_radiation_shield,
    build_upper_port,
    build_vacuum_vessel,
    build_vacuum_vessel_thermal_shield,
)
from matproplib.conditions import OperationalConditions

from bluemira_st.build_routines import SphericalReactor
from bluemira_st.params import BluemiraSTParams


class ReactorFactory:
    """A reactor creation wrapper with two components."""

    reactor_type: str
    reactor_config: ReactorConfig
    reactor: SphericalReactor | EUDEMO
    reference_eq: Equilibrium

    def reactor_factory(
        self, reactor_type: str, build_config: str | Path | dict
    ) -> SphericalReactor | EUDEMO:
        self.reactor_type = reactor_type
        match self.reactor_type:
            case "eudemo":
                self.reactor_config = ReactorConfig(build_config, EUDEMOReactorParams)
                self.reactor = self.create_eudemo()
            case "spherical_reactor":
                self.reactor_config = ReactorConfig(build_config, BluemiraSTParams)
                self.reactor = self.create_spherical_reactor()
        return self.reactor

    def build_reference_equilibrium(
        self, lcfs_coords=None, profiles=None
    ) -> Equilibrium:
        params_obj = self.reactor_config.params_for("free_boundary_equilibrium")
        config = self.reactor_config.config_for("free_boundary_equilibrium")

        if lcfs_coords is not None and profiles is not None:
            self.reactor.equilibria = EquilibriumManager()
            return self.reactor.build_reference_equilibrium(
                params_obj,
                config,
                self.reactor.equilibria,
                lcfs_coords,
                profiles,
            )
        return self.reactor.build_reference_equilibrium(params_obj.global_params, config)

    def build_plasma(self):
        params = self.reactor_config.params_for("free_boundary_equilibrium")
        config = self.reactor_config.config_for("free_boundary_equilibrium")
        self.reactor.plasma = self.reactor.build_plasma(
            params, config, self.reference_eq
        )

    def radial_build(self):
        params = self.reactor_config.params_for("radial_build")
        config = self.reactor_config.config_for("radial_build")
        self.reactor.radial_build(params.global_params, config)

    def build_pf_coils(self, pf_coil_keep_out_zones=None):
        params = self.reactor_config.params_for("pf_coils")
        config = self.reactor_config.config_for("pf_coils")
        if pf_coil_keep_out_zones is not None:
            self.reactor.pf_coil = self.reactor.build_pf_coils(
                params,
                config,
                self.reactor.equilibria,
                self.reactor.tf_coils.xz_outer_boundary,
                pf_coil_keep_out_zones,
            )
        else:
            self.reactor.pf_coil = self.reactor.build_pf_coils(
                params,
                config,
                self.reference_eq.coilset,
            )

    def build_tf_coils(self, vv_thermal_shield=None):
        params = self.reactor_config.params_for("tf_coils")
        config = self.reactor_config.config_for("tf_coils")
        if vv_thermal_shield is not None:
            self.reactor.tf_coils = self.reactor.build_tf_coils(
                params,
                config,
                self.reactor.plasma.lcfs(),
                vv_thermal_shield.xz_boundary,
            )
        else:
            self.reactor.tf_coils = self.reactor.build_tf_coils(
                params,
                config,
                self.reference_eq.coilset,
                interpolate_bspline(self.reference_eq.get_LCFS(), closed=True),
            )

    def build_blankets(self, ivc_shapes=None, r_inner_cut=None, cut_angle=None):

        if ivc_shapes is not None and r_inner_cut is not None and cut_angle is not None:
            params = self.reactor_config.params_for("blanket")
            config = self.reactor_config.config_for("blanket")
            self.reactor.blanket = build_blanket(
                params,
                config,
                ivc_shapes.inner_boundary,
                ivc_shapes.blanket_face,
                r_inner_cut,
                cut_angle,
            )
        else:
            self.reactor.blanket = self.reactor.build_bb(
                params,
                config,
                mat_name="BB_BZ_MATERIAL",
                ref_fbe=self.reference_eq,
            )

    def create_spherical_reactor(self) -> SphericalReactor:
        """Reactor function."""
        reactor = SphericalReactor(
            "Bluemira Spherical Tokamak Example",
            n_sectors=self.reactor_config.global_params.n_TF.value,
        )

        establish_material_cache([
            "bluemira_st.materials",
            "matproplib",
            Path(get_bluemira_root(), "examples", "design", "design_materials.py")
            .resolve()
            .as_posix(),
        ])

        self.radial_build()
        self.reference_eq = self.build_reference_equilibrium()
        self.build_plasma()
        self.build_pf_coils()
        self.build_tf_coils()
        self.build_blankets()

        self.reactor.inboard_shield = self.reactor.build_is(
            self.reactor_config.params_for("inboard_shield"),
            self.reactor_config.config_for("inboard_shield"),
            mat_name="EUROFER_MAT",
            ref_fbe=self.reference_eq,
        )
        # reactor.show_cad("xyz")
        # reactor.show_cad("xz")

        return reactor

    def create_eudemo(self) -> EUDEMO:
        reactor = EUDEMO(
            "EUDEMO", n_sectors=self.reactor_config.global_params.n_TF.value
        )

        establish_material_cache([
            "eudemo.materials",
            "eurofusion_materials.library",
            "matproplib",
        ])
        self.radial_build()

        lcfs_coords, profiles = run_designer(
            FixedEquilibriumDesigner,
            self.reactor_config.params_for("fixed_boundary_equilibrium"),
            self.reactor_config.config_for("fixed_boundary_equilibrium"),
        )

        lcfs_coords, profiles = run_designer(
            DummyFixedEquilibriumDesigner,
            self.reactor_config.params_for("dummy_fixed_boundary_equilibrium"),
            self.reactor_config.config_for("dummy_fixed_boundary_equilibrium"),
        )

        self.reference_eq = self.build_reference_equilibrium(
            lcfs_coords,
            profiles,
        )

        self.build_plasma()

        ivc_shapes = design_ivc(
            self.reactor_config.params_for("IVC").global_params,
            self.reactor_config.config_for("IVC"),
            equilibrium=self.reference_eq,
        )

        reactor.vacuum_vessel = build_vacuum_vessel(
            self.reactor_config.params_for("vacuum_vessel"),
            self.reactor_config.config_for("vacuum_vessel"),
            ivc_shapes.outer_boundary,
        )

        reactor.divertor = build_divertor(
            self.reactor_config.params_for("divertor"),
            self.reactor_config.config_for("divertor"),
            ivc_shapes.divertor_face,
        )

        upper_port_designer = UpperPortKOZDesigner(
            self.reactor_config.params_for("upper_port"),
            self.reactor_config.config_for("upper_port"),
            ivc_shapes.blanket_face,
        )
        upper_port_koz_xz, r_inner_cut, cut_angle = upper_port_designer.execute()

        self.build_blankets(
            ivc_shapes,
            r_inner_cut,
            cut_angle,
        )

        zero_d_params = ZeroDNeutronicsModel(self.reactor_config.global_params).run()

        self.reactor_config.global_params.update_from_frame(zero_d_params)
        if self.reactor_config.config_for("neutronics", "CSG").get("enabled", False):
            neutronics_csg = run_csg_neutronics(
                self.reactor_config.params_for("neutronics", "CSG").global_params,
                self.reactor_config.config_for("neutronics", "CSG"),
                blanket=reactor.blanket,
                vacuum_vessel=reactor.vacuum_vessel,
                ivc_shapes=ivc_shapes,
                eq=self.reference_eq,
                op_cond=OperationalConditions(temperature=298, pressure=101325),
            )
            if self.reactor_config.config_for("neutronics", "CSG")["show_data"]:
                reactor.neutronics.plot()
                bluemira_print(f"{reactor.neutronics}")
        else:
            neutronics_csg = None

        reactor.neutronics = NeutronicsManager(zero_d_params, neutronics_csg)

        vv_thermal_shield = build_vacuum_vessel_thermal_shield(
            self.reactor_config.params_for("thermal_shield"),
            self.reactor_config.config_for("thermal_shield", "VVTS"),
            self.reactor.vacuum_vessel.xz_boundary,
        )

        self.build_tf_coils(vv_thermal_shield)

        eq_port_designer = EquatorialPortKOZDesigner(
            self.reactor_config.params_for("equatorial_port"),
            self.reactor_config.config_for("equatorial_port"),
            x_ob=20.0,
        )

        eq_port_koz_xz = eq_port_designer.execute()

        (
            lp_duct_xz_void_space,
            lower_port_koz_xz,
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
        ) = LowerPortKOZDesigner(
            self.reactor_config.params_for("lower_port"),
            self.reactor_config.config_for("lower_port"),
            ivc_shapes.divertor_face,
            ivc_shapes.div_wall_join_pt,
            reactor.tf_coils.xz_outer_boundary,
        ).execute()

        self.build_pf_coils([
            upper_port_koz_xz,
            eq_port_koz_xz,
            lower_port_koz_xz,
        ])

        debug = [upper_port_koz_xz, eq_port_koz_xz, lower_port_koz_xz]
        debug.extend([reactor.tf_coils.xz_outer_boundary])
        debug.extend(reactor.pf_coils.xz_boundary)
        # I know there are clashes, I need to put in dynamic bounds on position opt to
        # include coil XS.
        # show_cad(debug)

        cryostat_thermal_shield = build_cryots(
            self.reactor_config.params_for("thermal_shield"),
            self.reactor_config.config_for("thermal_shield", "cryostat"),
            reactor.pf_coils.xz_boundary,
            reactor.tf_coils.xz_outer_boundary,
        )

        reactor.thermal_shield = assemble_thermal_shield(
            vv_thermal_shield, cryostat_thermal_shield
        )

        reactor.coil_structures = build_coil_structures(
            self.reactor_config.params_for("coil_structures"),
            self.reactor_config.config_for("coil_structures"),
            tf_coil_xz_face=reactor.tf_coils.xz_face,
            pf_coil_xz_wires=reactor.pf_coils.PF_xz_boundary,
            pf_coil_keep_out_zones=[
                upper_port_koz_xz,
                eq_port_koz_xz,
                lower_port_koz_xz,
            ],
        )

        reactor.cryostat = build_cryostat(
            self.reactor_config.params_for("cryostat"),
            self.reactor_config.config_for("cryostat"),
            cryostat_thermal_shield.xz_boundary,
        )

        reactor.radiation_shield = build_radiation_shield(
            self.reactor_config.params_for("radiation_shield"),
            self.reactor_config.config_for("radiation_shield"),
            reactor.cryostat.xz_boundary,
        )

        # Incorporate ports
        # TODO: Make potentially larger depending on where the PF
        # coils ended up. Warn if this isn't the case.

        ts_upper_port, vv_upper_port = build_upper_port(
            self.reactor_config.params_for("upper_port"),
            self.reactor_config.config_for("upper_port"),
            upper_port_koz_xz,
            reactor.pf_coils,
            cryostat_thermal_shield.xz_boundary,
        )
        ts_eq_port, vv_eq_port = build_equatorial_port(
            self.reactor_config.params_for("equatorial_port"),
            self.reactor_config.config_for("equatorial_port"),
            cryostat_thermal_shield.xz_boundary,
        )

        ts_lower_port, vv_lower_port = build_lower_port(
            self.reactor_config.params_for("lower_port"),
            self.reactor_config.config_for("lower_port"),
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
            reactor.cryostat.xz_boundary,
        )

        reactor.vacuum_vessel.add_ports(
            [vv_upper_port, vv_eq_port, vv_lower_port],
            n_TF=self.reactor_config.global_params.n_TF.value,
        )

        reactor.thermal_shield.add_ports(
            [ts_upper_port, ts_eq_port, ts_lower_port],
            n_TF=self.reactor_config.global_params.n_TF.value,
        )

        cr_plugs = build_cryostat_plugs(
            self.reactor_config.params_for("cryostat"),
            self.reactor_config.config_for("cryostat"),
            [ts_upper_port, ts_eq_port, ts_lower_port],
            reactor.cryostat.xz_boundary,
        )

        rs_plugs = build_radiation_plugs(
            self.reactor_config.params_for("radiation_shield"),
            self.reactor_config.config_for("radiation_shield"),
            cr_plugs,
            reactor.radiation_shield.xz_boundary,
        )

        reactor.cryostat.add_plugs(
            cr_plugs, n_TF=self.reactor_config.global_params.n_TF.value
        )

        reactor.radiation_shield.add_plugs(
            rs_plugs, n_TF=self.reactor_config.global_params.n_TF.value
        )

        reactor.neutronics.dagmc = run_dagmc_neutronics(
            reactor,
            self.reactor_config.params_for("neutronics", "DAGMC").global_params,
            self.reactor_config.config_for("neutronics", "DAGMC"),
            self.reference_eq,
        )

        sspc_solver = SteadyStatePowerCycleSolver(self.reactor_config.global_params)
        sspc_result = sspc_solver.execute()
        self.reactor_config.global_params.P_el_net.set_value(
            sspc_result["P_el_net"], "BLUEMIRA"
        )

        lcfs = ClosedFluxSurface(self.reference_eq.get_LCFS())

        self.reactor_config.global_params.V_p.set_value(lcfs.volume, "BLUEMIRA")


if __name__ == "__main__":
    # BUILD_CONFIG_FILE_PATH = Path(Path(__file__).parent, "config/config.json").resolve()
    BUILD_CONFIG_FILE_PATH = Path(
        Path(__file__).parent.parent.parent, "bluemira/eudemo/config/build_config.json"
    ).resolve()

    rf = ReactorFactory()
    reactor = rf.reactor_factory("eudemo", BUILD_CONFIG_FILE_PATH)
