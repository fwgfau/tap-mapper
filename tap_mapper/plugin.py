from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler import PassManager

from tap_mapper.tap_pass_grb import TAPMapping


class TAPMappingPlugin(PassManagerStagePlugin):
    """Plugin."""

    def pass_manager(self, pass_manager_config, optimization_level):
        """Returns the plugin pass manager."""
        pm = PassManager(
            [
                TAPMapping(
                    pass_manager_config.coupling_map,
                ),
                CheckMap(pass_manager_config.coupling_map),
            ]
        )
        return pm
