"""Check GTEx publication and request failures with tiny offline fixtures only."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import http.client
import io
import json
import os
from pathlib import Path
import socket
import ssl
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import call, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
with patch("urllib.request.urlopen", side_effect=AssertionError("No requests on import")), \
     patch("urllib.request.urlretrieve", side_effect=AssertionError("No downloads on import")), \
     patch.object(pd, "read_csv", side_effect=AssertionError("No research CSV reads")):
    import build_gtex_pilot as builder


class Response(io.BytesIO):
    def __init__(self, content, headers=None):
        super().__init__(content)
        self.headers = {} if headers is None else headers


def source_fixture():
    # Exactly nine metadata fields precede the established 30-tissue block.
    fields = {
        "transcript_id_gtex": ["tx-a", "tx-b"], "gene_id_gtex": ["gene-a", "gene-b"],
        "split": ["train", "test"], "chr": ["chr1", "chr2"], "strand": ["+", "-"],
        "start": [101, 301], "end": [200, 400], "5UTR": ["AA", "CC"], "3UTR": ["TT", "GG"],
    }
    fields.update({tissue: [index / 4, index / 4 + 0.5]
                   for index, tissue in enumerate(builder.GTEX_TISSUES)})
    fields.update({"RNA": ["ATGGAA", "ATGTAA"], "CDS": ["ATGGAA", "ATGTAA"], "Protein": ["ME", "M"]})
    return pd.DataFrame(fields)


def pilot_fixture():
    table = source_fixture().assign(TSS=[100, 399], DNA=["ACGT", "TGCA"])
    table = table.rename(columns={"transcript_id_gtex": "transcript_id", "gene_id_gtex": "gene_id"})
    columns = ["transcript_id", "gene_id", "split", "chr", "strand", "TSS", "DNA", "RNA",
               "5UTR", "CDS", "3UTR", "Protein", *builder.GTEX_TISSUES]
    return table[columns]


def json_response(payload):
    return Response(json.dumps(payload).encode())


def http_error(code, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(builder.ENSEMBL_URL, code, "fixture", headers, io.BytesIO(b"error"))


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.resources.enter_context(patch.object(builder, "DOWNLOAD_DIR", self.root / "source"))
        self.resources.enter_context(patch.object(builder, "OUTPUT", self.root / "pilot.csv"))
        self.urlopen = self.resources.enter_context(patch.object(
            builder.urllib.request, "urlopen", side_effect=AssertionError("Unconfigured network request")))
        self.resources.enter_context(patch("urllib.request.urlretrieve", side_effect=AssertionError("No downloads")))
        self.resources.enter_context(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.resources.enter_context(patch("socket.create_connection", side_effect=AssertionError("No network")))
        self.resources.enter_context(patch("mbf.encoders.load_encoder", side_effect=AssertionError("No models")))
        self.resources.enter_context(patch("mbf.encoders.load_tokenizer", side_effect=AssertionError("No models")))
        self.sleep = self.resources.enter_context(patch.object(builder.time, "sleep"))
        self.resources.enter_context(redirect_stdout(io.StringIO()))
        self.resources.enter_context(redirect_stderr(io.StringIO()))

    def source_response(self, payload, headers=None):
        response = Response(payload, headers)
        self.urlopen.side_effect = [response]
        return response

    def assert_no_temporary_files(self):
        self.assertEqual(list(self.root.rglob("*.tmp")), [])


class SourceTableTests(OfflineTests):
    def test_valid_download_checks_transport_and_publishes_from_a_sibling(self):
        frame = source_fixture()
        payload = frame.to_csv(index=False).encode()
        response = self.source_response(payload, {"Content-Length": str(len(payload))})
        final = builder.DOWNLOAD_DIR / "GTEx_final.csv"
        original_replace = os.replace

        def check_publication(temporary, destination):
            self.assertEqual(temporary.parent, final.parent)
            self.assertEqual(destination, final)
            self.assertFalse(final.exists())
            self.assertEqual(temporary.read_bytes(), payload)
            original_replace(temporary, destination)

        with patch.object(builder.os, "replace", side_effect=check_publication) as replace:
            actual = builder.source_table()
        pd.testing.assert_frame_equal(actual, frame)
        self.assertEqual(final.read_bytes(), payload)
        self.urlopen.assert_called_once_with(builder.SOURCE_URL, timeout=120)
        replace.assert_called_once()
        self.assertTrue(response.closed)
        self.assert_no_temporary_files()

    def test_download_without_content_length_still_validates_schema(self):
        frame = source_fixture()
        self.source_response(frame.to_csv(index=False).encode())
        pd.testing.assert_frame_equal(builder.source_table(), frame)
        self.assert_no_temporary_files()

    def test_incorrect_or_invalid_content_length_never_publishes(self):
        payload = source_fixture().to_csv(index=False).encode()
        for declared in (str(len(payload) - 1), str(len(payload) + 1), "-1", "invalid"):
            with self.subTest(declared=declared):
                response = self.source_response(payload, {"Content-Length": declared})
                with self.assertRaises(ValueError):
                    builder.source_table()
                self.assertFalse((builder.DOWNLOAD_DIR / "GTEx_final.csv").exists())
                self.assertTrue(response.closed)
                self.assert_no_temporary_files()

    def test_interrupted_download_preserves_the_original_error(self):
        failure = TimeoutError("fixture download interrupted")

        class InterruptedResponse(Response):
            def read(self, size=-1):
                if self.tell():
                    raise failure
                return super().read(8)

        response = InterruptedResponse(source_fixture().to_csv(index=False).encode())
        self.urlopen.side_effect = [response]
        with self.assertRaises(TimeoutError) as caught:
            builder.source_table()
        self.assertIs(caught.exception, failure)
        self.assertFalse((builder.DOWNLOAD_DIR / "GTEx_final.csv").exists())
        self.assertTrue(response.closed)
        self.assert_no_temporary_files()

    def test_invalid_source_fields_tissue_order_and_empty_table_never_publish(self):
        frame = source_fixture()
        swapped = list(frame.columns)
        swapped[9], swapped[10] = swapped[10], swapped[9]
        cases = (frame.drop(columns="Protein"), frame[swapped],
                 frame.rename(columns={builder.GTEX_TISSUES[0]: "Unknown tissue"}), frame.iloc[:0])
        for invalid in cases:
            with self.subTest(columns=list(invalid.columns), rows=len(invalid)):
                self.source_response(invalid.to_csv(index=False).encode())
                with self.assertRaises(ValueError):
                    builder.source_table()
                self.assertFalse((builder.DOWNLOAD_DIR / "GTEx_final.csv").exists())
                self.assert_no_temporary_files()

    def test_existing_valid_and_invalid_caches_are_read_without_network(self):
        path = builder.DOWNLOAD_DIR / "GTEx_final.csv"
        path.parent.mkdir(parents=True)
        frame = source_fixture()
        frame.to_csv(path, index=False)
        before = path.read_bytes()
        pd.testing.assert_frame_equal(builder.source_table(), frame)
        self.assertEqual(path.read_bytes(), before)
        frame.drop(columns="CDS").to_csv(path, index=False)
        invalid = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            builder.source_table()
        self.assertEqual(path.read_bytes(), invalid)
        self.urlopen.assert_not_called()

    def test_failed_source_replacement_does_not_publish_a_partial_file(self):
        self.source_response(source_fixture().to_csv(index=False).encode())
        failure = PermissionError("fixture replacement denied")
        with patch.object(builder.os, "replace", side_effect=failure):
            with self.assertRaises(PermissionError) as caught:
                builder.source_table()
        self.assertIs(caught.exception, failure)
        self.assertFalse((builder.DOWNLOAD_DIR / "GTEx_final.csv").exists())
        self.assert_no_temporary_files()

    def test_source_http_error_response_is_closed(self):
        failure = http_error(404)
        response = failure.fp
        self.urlopen.side_effect = failure
        with self.assertRaises(urllib.error.HTTPError) as caught:
            builder.source_table()
        self.assertIs(caught.exception, failure)
        self.assertTrue(response.closed)
        self.sleep.assert_not_called()
        self.assert_no_temporary_files()

    def test_temporary_cleanup_failure_does_not_hide_the_download_failure(self):
        failure = TimeoutError("fixture connection timeout")
        self.urlopen.side_effect = failure
        with patch.object(Path, "unlink", side_effect=PermissionError("fixture cleanup denied")):
            with self.assertRaises(TimeoutError) as caught:
                builder.source_table()
        self.assertIs(caught.exception, failure)
        if hasattr(failure, "add_note"):
            self.assertTrue(any("fixture cleanup denied" in note for note in failure.__notes__))
        self.assertFalse((builder.DOWNLOAD_DIR / "GTEx_final.csv").exists())


class PilotPublicationTests(OfflineTests):
    def test_successful_pilot_replaces_old_file_only_after_validation(self):
        frame = pilot_fixture()
        builder.OUTPUT.write_bytes(b"previous complete output")
        original_replace = os.replace

        def check_publication(temporary, destination):
            self.assertEqual(builder.OUTPUT.read_bytes(), b"previous complete output")
            self.assertEqual(temporary.parent, builder.OUTPUT.parent)
            pd.testing.assert_frame_equal(pd.read_csv(temporary), frame)
            original_replace(temporary, destination)

        with patch.object(builder.os, "replace", side_effect=check_publication):
            builder._write_pilot(frame, builder.OUTPUT)
        pd.testing.assert_frame_equal(pd.read_csv(builder.OUTPUT), frame)
        self.assert_no_temporary_files()

    def test_write_failure_keeps_previous_output_and_original_error(self):
        builder.OUTPUT.write_bytes(b"previous complete output")
        failure = OSError("fixture write interrupted")

        def interrupted_write(frame, destination, **kwargs):
            Path(destination).write_text("partial", encoding="utf-8")
            raise failure

        with patch.object(pd.DataFrame, "to_csv", autospec=True, side_effect=interrupted_write):
            with self.assertRaises(OSError) as caught:
                builder._write_pilot(pilot_fixture(), builder.OUTPUT)
        self.assertIs(caught.exception, failure)
        self.assertEqual(builder.OUTPUT.read_bytes(), b"previous complete output")
        self.assert_no_temporary_files()

    def test_incomplete_csv_keeps_previous_output(self):
        frame = pilot_fixture()
        builder.OUTPUT.write_bytes(b"previous complete output")
        header = frame.iloc[:0].to_csv(index=False)

        def incomplete_write(frame, destination, **kwargs):
            Path(destination).write_text(header, encoding="utf-8")

        with patch.object(pd.DataFrame, "to_csv", autospec=True, side_effect=incomplete_write):
            with self.assertRaisesRegex(ValueError, "row count"):
                builder._write_pilot(frame, builder.OUTPUT)
        self.assertEqual(builder.OUTPUT.read_bytes(), b"previous complete output")
        self.assert_no_temporary_files()

    def test_failed_replacement_keeps_previous_output(self):
        builder.OUTPUT.write_bytes(b"previous complete output")
        failure = PermissionError("fixture output replacement denied")
        with patch.object(builder.os, "replace", side_effect=failure):
            with self.assertRaises(PermissionError) as caught:
                builder._write_pilot(pilot_fixture(), builder.OUTPUT)
        self.assertIs(caught.exception, failure)
        self.assertEqual(builder.OUTPUT.read_bytes(), b"previous complete output")
        self.assert_no_temporary_files()


class EnsemblRequestTests(OfflineTests):
    def test_batches_are_bounded_deduplicated_and_preserve_row_lookups(self):
        unique = [f"1:{index + 1}..{index + 4}:1" for index in range(52)]
        regions = unique[:3] + [unique[0]] + unique[3:] + [unique[-1]]
        before = regions.copy()
        requested = []

        def respond(request, timeout):
            self.assertEqual(request.full_url, builder.ENSEMBL_URL)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(timeout, 120)
            batch = json.loads(request.data)["regions"]
            requested.append(batch)
            return json_response([{"query": region, "seq": "acgtn"} for region in reversed(batch)])

        self.urlopen.side_effect = respond
        sequences = builder.fetch_regions(regions)
        self.assertEqual(requested, [unique[:50], unique[50:]])
        self.assertEqual(sequences, dict.fromkeys(unique, "ACGTN"))
        self.assertEqual([sequences[region] for region in regions], ["ACGTN"] * len(regions))
        self.assertEqual(regions, before)
        self.assertEqual(self.sleep.call_args_list, [call(0.3), call(0.3)])

    def test_empty_requests_do_not_connect_or_delay(self):
        self.assertEqual(builder.fetch_regions([]), {})
        self.urlopen.assert_not_called()
        self.sleep.assert_not_called()

    def test_defined_transient_http_and_connection_failures_retry_then_succeed(self):
        failures = [http_error(code) for code in (429, 500, 502, 503, 504)]
        failures += [TimeoutError("timeout"), ConnectionResetError("reset"),
                     urllib.error.URLError(TimeoutError("timeout")),
                     urllib.error.URLError(ConnectionRefusedError("refused")),
                     urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "temporary DNS"))]
        for failure in failures:
            with self.subTest(failure=repr(failure)):
                self.urlopen.reset_mock()
                self.sleep.reset_mock()
                self.urlopen.side_effect = [failure, json_response([{"query": "r", "seq": "ac"}])]
                self.assertEqual(builder.fetch_regions(["r"]), {"r": "AC"})
                self.assertEqual(self.urlopen.call_count, 2)
                self.assertEqual(self.sleep.call_args_list, [call(5), call(0.3)])

    def test_exhaustion_preserves_last_cause_without_a_final_delay(self):
        for failures in ([TimeoutError(f"attempt-{i}") for i in range(5)],
                         [http_error(503) for _ in range(5)]):
            with self.subTest(failure=type(failures[0]).__name__):
                self.urlopen.reset_mock()
                self.sleep.reset_mock()
                self.urlopen.side_effect = failures
                with self.assertRaisesRegex(RuntimeError, "five attempts") as caught:
                    builder.fetch_regions(["r", "next-region"])
                self.assertIs(caught.exception.__cause__, failures[-1])
                self.assertIn("batch size 2, first region 'r'", str(caught.exception))
                self.assertEqual(self.urlopen.call_count, 5)
                self.assertEqual(self.sleep.call_args_list, [call(5), call(10), call(15), call(20)])

    def test_fractional_retry_after_is_honored_within_the_cap(self):
        for value in ("0", "1.25", "120"):
            with self.subTest(retry_after=value):
                self.sleep.reset_mock()
                self.urlopen.side_effect = [http_error(429, value),
                                           json_response([{"query": "r", "seq": "ac"}])]
                builder.fetch_regions(["r"])
                self.assertEqual(self.sleep.call_args_list, [call(float(value)), call(0.3)])

    def test_http_error_responses_close_on_retry_permanent_failure_and_exhaustion(self):
        for failures, succeeds in (([http_error(503)], True), ([http_error(404)], False),
                                   ([http_error(503) for _ in range(5)], False)):
            with self.subTest(attempts=len(failures), succeeds=succeeds):
                responses = [failure.fp for failure in failures]
                effects = list(failures)
                if succeeds:
                    effects.append(json_response([{"query": "r", "seq": "AC"}]))
                self.urlopen.side_effect = effects
                if succeeds:
                    builder.fetch_regions(["r"])
                else:
                    with self.assertRaises((urllib.error.HTTPError, RuntimeError)):
                        builder.fetch_regions(["r"])
                self.assertTrue(all(response.closed for response in responses))

    def test_invalid_or_excessive_retry_after_stops_without_sleeping(self):
        for value in ("120.01", "invalid", "-1", "NaN", "inf"):
            with self.subTest(retry_after=value):
                self.urlopen.reset_mock()
                failure = http_error(429, value)
                self.urlopen.side_effect = failure
                with self.assertRaisesRegex(RuntimeError, "Retry-After") as caught:
                    builder.fetch_regions(["r"])
                self.assertIs(caught.exception.__cause__, failure)
                self.assertEqual(self.urlopen.call_count, 1)
                self.sleep.assert_not_called()

    def test_permanent_http_certificate_protocol_and_other_errors_are_not_retried(self):
        failures = [http_error(code) for code in (400, 404, 501)]
        failures += [urllib.error.URLError(ssl.SSLCertVerificationError(1, "bad certificate")),
                     urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "unknown hostname")),
                     urllib.error.URLError("unsupported protocol"), http.client.BadStatusLine("bad status"),
                     ValueError("fixture programming error")]
        for failure in failures:
            with self.subTest(failure=repr(failure)):
                self.urlopen.reset_mock()
                self.urlopen.side_effect = failure
                with self.assertRaises(type(failure)) as caught:
                    builder.fetch_regions(["r"])
                self.assertIs(caught.exception, failure)
                self.assertEqual(self.urlopen.call_count, 1)
                self.sleep.assert_not_called()

    def test_invalid_json_and_whole_batch_schema_fail_without_retry(self):
        payloads = [{}, [None], [], [{"query": "r", "seq": None}],
                    [{"query": "r", "seq": " "}], [{"seq": "AC"}],
                    [{"query": "unexpected", "seq": "AC"}],
                    [{"query": "r", "seq": "AC"}, {"query": "r", "seq": "GT"}]]
        responses = [Response(b"[")] + [json_response(payload) for payload in payloads]
        for response in responses:
            with self.subTest(response=response.getvalue()):
                self.urlopen.reset_mock()
                self.urlopen.side_effect = [response]
                with self.assertRaises(ValueError):
                    builder.fetch_regions(["r"])
                self.assertEqual(self.urlopen.call_count, 1)
                self.assertTrue(response.closed)
                self.sleep.assert_not_called()

    def test_missing_region_in_later_batch_does_not_return_partial_results(self):
        regions = [f"region-{index}" for index in range(51)]
        self.urlopen.side_effect = [json_response([{"query": r, "seq": "AC"} for r in regions[:50]]),
                                   json_response([])]
        with self.assertRaisesRegex(ValueError, "missing requested regions"):
            builder.fetch_regions(regions)
        self.assertEqual(self.urlopen.call_count, 2)
        self.assertEqual(self.sleep.call_args_list, [call(0.3)])


if __name__ == "__main__":
    unittest.main()
