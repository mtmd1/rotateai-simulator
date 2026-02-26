'''
tests/test_main.py
Unit tests for simulator/__main__.py

Created: 2026-02-26
 Author: Maxence Morel Dierckx
'''
import sys
import signal
import pytest
from unittest.mock import patch, MagicMock, call
from types import SimpleNamespace
from simulator.__main__ import handle_shutdown, simulate, main


# MARK: handle_shutdown

class TestHandleShutdown:
    '''Tests for the signal handler.'''

    def test_exits_on_sigint(self):
        with pytest.raises(SystemExit):
            handle_shutdown(signal.SIGINT, None)

    def test_exits_on_sigterm(self):
        with pytest.raises(SystemExit):
            handle_shutdown(signal.SIGTERM, None)

    def test_exit_code_is_one(self):
        with pytest.raises(SystemExit) as exc_info:
            handle_shutdown(signal.SIGINT, None)
        assert exc_info.value.code == 1

    def test_prints_signal_name(self, capsys):
        with pytest.raises(SystemExit):
            handle_shutdown(signal.SIGINT, None)
        assert 'SIGINT' in capsys.readouterr().err

    def test_prints_sigterm_name(self, capsys):
        with pytest.raises(SystemExit):
            handle_shutdown(signal.SIGTERM, None)
        assert 'SIGTERM' in capsys.readouterr().err


# MARK: simulate

class TestSimulate:
    '''Tests for the simulate orchestrator.'''

    def _make_args(self, config='config.toml', binary='path/to/inference', data='data/'):
        return SimpleNamespace(config=config, binary=binary, data=data)

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_loads_config_data_simulator(self, MockConfig, MockData, MockSimulator, mock_save):
        MockData.return_value.__iter__ = MagicMock(return_value=iter([]))
        MockData.return_value.__len__ = MagicMock(return_value=0)
        args = self._make_args()
        simulate(args)

        MockConfig.assert_called_once_with('config.toml')
        MockData.assert_called_once_with('data/')
        MockSimulator.assert_called_once_with('path/to/inference')

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_runs_each_batch(self, MockConfig, MockData, MockSimulator, mock_save):
        batch_0, batch_1 = {'Mw': 'b0'}, {'Mw': 'b1'}
        MockData.return_value.__iter__ = MagicMock(return_value=iter([batch_0, batch_1]))
        MockData.return_value.__len__ = MagicMock(return_value=2)
        mock_sim = MockSimulator.return_value

        args = self._make_args()
        simulate(args)

        assert mock_sim.run.call_count == 2

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_saves_report_per_batch(self, MockConfig, MockData, MockSimulator, mock_save):
        batch_0, batch_1 = {'Mw': 'b0'}, {'Mw': 'b1'}
        MockData.return_value.__iter__ = MagicMock(return_value=iter([batch_0, batch_1]))
        MockData.return_value.__len__ = MagicMock(return_value=2)

        args = self._make_args()
        simulate(args)

        assert mock_save.call_count == 2

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_report_name_derives_from_binary(self, MockConfig, MockData, MockSimulator, mock_save):
        MockData.return_value.__iter__ = MagicMock(return_value=iter([{'Mw': 'b0'}]))
        MockData.return_value.__len__ = MagicMock(return_value=1)

        args = self._make_args(binary='some/nested/path/inference')
        simulate(args)

        name_arg = mock_save.call_args_list[0][0][0]
        assert name_arg == 'simreport_inference_0'

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_report_index_increments(self, MockConfig, MockData, MockSimulator, mock_save):
        batches = [{'i': 0}, {'i': 1}, {'i': 2}]
        MockData.return_value.__iter__ = MagicMock(return_value=iter(batches))
        MockData.return_value.__len__ = MagicMock(return_value=3)

        args = self._make_args(binary='inference')
        simulate(args)

        names = [c[0][0] for c in mock_save.call_args_list]
        assert names == ['simreport_inference_0', 'simreport_inference_1', 'simreport_inference_2']

    @patch('simulator.__main__.save_report')
    @patch('simulator.__main__.Simulator')
    @patch('simulator.__main__.Data')
    @patch('simulator.__main__.Config')
    def test_prints_done(self, MockConfig, MockData, MockSimulator, mock_save, capsys):
        MockData.return_value.__iter__ = MagicMock(return_value=iter([]))
        MockData.return_value.__len__ = MagicMock(return_value=0)

        simulate(self._make_args())
        assert 'Done.' in capsys.readouterr().out


# MARK: main

class TestMain:
    '''Tests for the CLI entry point.'''

    @patch('simulator.__main__.simulate')
    def test_registers_signal_handlers(self, mock_simulate):
        with patch('simulator.__main__.signal.signal') as mock_signal, \
             patch('simulator.__main__.argparse.ArgumentParser') as MockParser:
            MockParser.return_value.parse_args.return_value = SimpleNamespace(func=mock_simulate)
            main()

        calls = mock_signal.call_args_list
        sig_nums = [c[0][0] for c in calls]
        assert signal.SIGINT in sig_nums
        assert signal.SIGTERM in sig_nums

    @patch('simulator.__main__.simulate')
    def test_requires_config_arg(self, mock_simulate):
        with patch('sys.argv', ['simulate', '--binary', 'b', '--data', 'd']):
            with pytest.raises(SystemExit):
                main()

    @patch('simulator.__main__.simulate')
    def test_requires_binary_arg(self, mock_simulate):
        with patch('sys.argv', ['simulate', '--config', 'c', '--data', 'd']):
            with pytest.raises(SystemExit):
                main()

    @patch('simulator.__main__.simulate')
    def test_requires_data_arg(self, mock_simulate):
        with patch('sys.argv', ['simulate', '--config', 'c', '--binary', 'b']):
            with pytest.raises(SystemExit):
                main()

    @patch('simulator.__main__.simulate')
    def test_dispatches_to_simulate(self, mock_simulate):
        with patch('sys.argv', ['simulate', '-c', 'conf.toml', '-b', 'bin', '-d', 'data/']):
            main()
        mock_simulate.assert_called_once()

    @patch('simulator.__main__.simulate')
    def test_args_passed_correctly(self, mock_simulate):
        with patch('sys.argv', ['simulate', '-c', 'my.toml', '-b', 'my_bin', '-d', 'my_data/']):
            main()
        args = mock_simulate.call_args[0][0]
        assert args.config == 'my.toml'
        assert args.binary == 'my_bin'
        assert args.data == 'my_data/'
