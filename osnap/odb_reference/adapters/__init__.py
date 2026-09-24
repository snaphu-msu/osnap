"""Native format registry. No simulation-code installation is needed to import."""
from ..errors import FormatError
from .common import Adapter
from .flash import FLASHAdapter, STIRAdapter
from .stellar import MESAAdapter, KEPLERAdapter
from .skynet import SkyNetAdapter
from .radiation import SNECAdapter, TARDISAdapter

ADAPTERS = {a.format: a for a in (FLASHAdapter(), STIRAdapter(), MESAAdapter(), KEPLERAdapter(),
                                 SkyNetAdapter(), SNECAdapter(), TARDISAdapter())}


def read_native(source, *, format, **options):
    if format == "auto":
        matches = [a for a in ADAPTERS.values() if a.detect(source)]
        if len(matches) != 1:
            raise FormatError(f"Native detection is ambiguous or unsupported: {[a.format for a in matches]}; specify format")
        adapter = matches[0]
    else:
        try:
            adapter = ADAPTERS[format.lower()]
        except KeyError:
            raise FormatError(f"Unknown native format {format!r}; supported: {list(ADAPTERS)}") from None
    dataset = adapter.read(source, **options)
    dataset.provenance.append({"operation": "native_import", "format": adapter.format, "report": dataset.report.to_dict()})
    return dataset


def export_native(dataset, target, *, format, selection, **options):
    try:
        adapter = ADAPTERS[format.lower()]
    except KeyError:
        raise FormatError(f"Unknown native export format {format!r}") from None
    if isinstance(selection, tuple) and len(selection) == 2:
        selection = dataset.series[selection[0]].snapshot(selection[1])
    return adapter.write(target, selection, **options)


__all__ = ["Adapter", "ADAPTERS", "read_native", "export_native"]
