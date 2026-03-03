/**
 * Draw.io Plugin: Power Block Diagram Current Calculator
 *
 * Automatically back-calculates load currents through a power distribution tree
 * in real-time as the diagram is built. Supports switching regulators, LDOs,
 * AC/DC rectifiers, filters, power switches, current sums, and loads.
 *
 * Works with both:
 *   - New format: properties on parent group objects, displayed via %placeholder%
 *   - Old format: values stored in child cell labels with type attributes
 *
 * Usage: Load this plugin in draw.io via Extras > Plugins.
 *        See README for detailed instructions.
 */
Draw.loadPlugin(function(ui) {
	var graph = ui.editor.graph;
	var model = graph.getModel();

	var ROOT_TYPES = ['sw_reg', 'ldo', 'acdc', 'source', 'load', 'filter', 'isum', 'pwr_sw'];
	var isRecalculating = false;
	var recalcTimer = null;

	// -------------------------------------------------------------------------
	// Value parsing (mirrors Python extract_and_convert)
	// -------------------------------------------------------------------------

	function removeHtmlTags(s) {
		return s.replace(/<[^>]*>/g, '').trim();
	}

	/**
	 * Parse a value string like "3.3V", "500mA", "85%", "1V2", "24VAC"
	 * into a floating-point number in base units (V, A, ratio).
	 */
	function parseValue(s) {
		if (typeof s !== 'string') return NaN;
		s = removeHtmlTags(s);
		if (!s || s.indexOf('?') >= 0) return NaN;

		var number = s.match(/[-+]?\d*\.?\d+/);
		if (!number) return NaN;

		var value = parseFloat(number[0]);
		if (s.charAt(0) === '-') value = -value;

		if (s.slice(-2) === 'mA') return value / 1e3;
		if (s.slice(-2) === 'uA') return value / 1e6;
		if (s.slice(-2) === 'mV') return value / 1e3;
		if (s.charAt(s.length - 1) === '%') return value / 1e2;

		if (s.indexOf('V') >= 0) {
			var vStr = s;
			if (s.slice(-2) === 'AC') vStr = s.slice(0, -2);
			var parts = vStr.split('V');
			if (parts.length === 2 && parts[1] !== '') {
				value = parseFloat(parts[0]) + parseFloat(parts[1]) / Math.pow(10, parts[1].length);
			}
		}

		return value;
	}

	function formatCurrent(amps) {
		if (amps >= 0.1) {
			return (Math.round(amps * 100) / 100) + 'A';
		}
		return Math.round(amps * 1000) + 'mA';
	}

	function formatVoltage(volts) {
		if (volts >= 0.1) {
			return (Math.round(volts * 100) / 100) + 'V';
		}
		return Math.round(volts * 1000) + 'mV';
	}

	// -------------------------------------------------------------------------
	// Cell helpers
	// -------------------------------------------------------------------------

	/** Return the custom `type` attribute if the cell is an <object> node. */
	function getCellType(cell) {
		if (cell && cell.value && typeof cell.value === 'object' && cell.value.nodeType) {
			return cell.getAttribute('type') || null;
		}
		return null;
	}

	/** Find all cells whose type is one of ROOT_TYPES. */
	function getComponents() {
		var components = {};
		var cells = model.cells;
		for (var id in cells) {
			if (cells.hasOwnProperty(id)) {
				var type = getCellType(cells[id]);
				if (type !== null && ROOT_TYPES.indexOf(type) >= 0) {
					components[id] = cells[id];
				}
			}
		}
		return components;
	}

	/** Walk up the parent chain from `cell` to find its component group. */
	function getGroupParent(cell, components) {
		if (!cell) return null;
		if (components[cell.id]) return cell;

		var parent = model.getParent(cell);
		if (!parent || parent.id === '0' || parent.id === '1') return null;
		if (components[parent.id]) return parent;
		return getGroupParent(parent, components);
	}

	/**
	 * Read a property value from a component.
	 *   New format – property stored on the group <object> itself.
	 *   Old format – value in the label of a child <object> with matching type.
	 */
	function readProperty(componentCell, propName) {
		// New format
		var val = componentCell.getAttribute(propName);
		if (val !== null && val !== undefined && val.charAt(0) !== '%') {
			return val;
		}

		// Old format – search children
		var childCount = model.getChildCount(componentCell);
		for (var i = 0; i < childCount; i++) {
			var child = model.getChildAt(componentCell, i);
			if (getCellType(child) === propName) {
				var label = child.getAttribute('label');
				if (label) return label;
			}
		}
		return null;
	}

	/**
	 * Write a calculated value back to the diagram.
	 * Returns true if the value actually changed.
	 */
	function writeProperty(componentCell, propName, newValue) {
		// New format – property on the group object
		var existing = componentCell.getAttribute(propName);
		if (existing !== null && existing !== undefined) {
			if (existing === newValue) return false;
			model.execute(new mxCellAttributeChange(componentCell, propName, newValue));
			return true;
		}

		// Old format – child cell label
		var childCount = model.getChildCount(componentCell);
		for (var i = 0; i < childCount; i++) {
			var child = model.getChildAt(componentCell, i);
			if (getCellType(child) === propName) {
				var curLabel = child.getAttribute('label');
				if (curLabel === newValue) return false;
				model.execute(new mxCellAttributeChange(child, 'label', newValue));
				return true;
			}
		}
		return false;
	}

	// -------------------------------------------------------------------------
	// Load multiplier  (e.g. "USB Connector x4" → multiplier of 4)
	// -------------------------------------------------------------------------

	function getLoadMultiplier(name) {
		if (!name) return 1;
		var m = name.match(/x(\d+)/);
		return m ? parseInt(m[1], 10) : 1;
	}

	/**
	 * Extract XOR group from a name:
	 *   "WiFi XOR1A" → "1"  (paired)
	 *   "Heater XOR3" → "3"  (rail-exclusive, checked separately)
	 *   "Motor XOR"   → "_global" (global-exclusive)
	 */
	function getXorGroup(name) {
		if (!name) return null;
		// Bare XOR (no number) = global exclusive
		if (/\bXOR\b/i.test(name) && !/XOR\d/i.test(name)) return '_global';
		// Numbered XOR group
		var m = name.match(/XOR(\d+)/i);
		return m ? m[1] : null;
	}

	/** Check if XOR tag has no letter suffix, e.g. "XOR3" vs "XOR3A" */
	function isRailExclusive(name) {
		if (!name) return false;
		var m = name.match(/XOR(\d+)([A-Z])?/i);
		if (!m) return false;
		return !m[2]; // no letter = rail exclusive
	}

	// -------------------------------------------------------------------------
	// Core calculation  (mirrors Python calculate_missing_values)
	// -------------------------------------------------------------------------

	function recalculate() {
		if (isRecalculating) return;
		isRecalculating = true;

		try {
			var components = getComponents();
			if (Object.keys(components).length === 0) return;

			// --- Build supply map (consumer → supplier) from edges -----------
			var supplyMap = {};
			var cells = model.cells;

			for (var id in cells) {
				if (!cells.hasOwnProperty(id)) continue;
				var cell = cells[id];
				if (model.isEdge(cell) && cell.source && cell.target) {
					var srcGroup = getGroupParent(cell.source, components);
					var tgtGroup = getGroupParent(cell.target, components);
					if (srcGroup && tgtGroup && srcGroup.id !== tgtGroup.id) {
						supplyMap[tgtGroup.id] = srcGroup.id;
					}
				}
			}

			// --- Build node graph --------------------------------------------
			var nodes = {};
			for (var cid in components) {
				if (!components.hasOwnProperty(cid)) continue;
				var comp = components[cid];
				var type = getCellType(comp);

				var vout        = parseValue(readProperty(comp, 'vout'));
				var efficiency  = parseValue(readProperty(comp, 'efficiency'));
				var loadName    = readProperty(comp, 'load_name');

				// Only read load_current from the diagram for 'load' components.
				// All other types must be recalculated from children
				// every time, otherwise stale values persist after edits.
				var loadCurrent = NaN;
				if (type === 'load') {
					loadCurrent = parseValue(readProperty(comp, 'load_current'));
					if (loadName) {
						var mult = getLoadMultiplier(loadName);
						if (mult > 1 && !isNaN(loadCurrent)) {
							loadCurrent *= mult;
						}
					}
				}

				nodes[cid] = {
					id: cid,
					type: type,
					vout: vout,
					current: loadCurrent,
					efficiency: isNaN(efficiency) ? 1.0 : efficiency,
					xorGroup: getXorGroup(loadName),
					railXor: isRailExclusive(loadName),
					normalCurrent: 0,
					xorCurrents: {},
					children: [],
					supplierId: supplyMap[cid] || null
				};
			}

			// Link children
			for (var nid in nodes) {
				if (!nodes.hasOwnProperty(nid)) continue;
				var sid = nodes[nid].supplierId;
				if (sid && nodes[sid]) {
					nodes[sid].children.push(nodes[nid]);
				}
			}


			// Initialize XOR tracking for leaf loads
			for (var lid in nodes) {
				if (!nodes.hasOwnProperty(lid)) continue;
				var leaf = nodes[lid];
				if (leaf.children.length === 0 && !isNaN(leaf.current)) {
					if (leaf.railXor) {
						// Rail-exclusive: parent handles via railXor flag
						leaf.normalCurrent = 0;
					} else if (leaf.xorGroup) {
						leaf.normalCurrent = 0;
						leaf.xorCurrents[leaf.xorGroup] = leaf.current;
					} else {
						leaf.normalCurrent = leaf.current;
					}
				}
			}
			// --- Propagate voltages top-down ---------------------------------
			function propagateVout(node) {
				for (var i = 0; i < node.children.length; i++) {
					var child = node.children[i];
					if (!isNaN(node.vout)) {
						if (['pwr_sw', 'filter', 'isum'].indexOf(child.type) >= 0) {
							child.vout = node.vout;
						} else if (child.type === 'acdc' && isNaN(child.vout)) {
							child.vout = node.vout * Math.SQRT2 - 1;
						}
					}
					propagateVout(child);
				}
			}

			// --- Calculate currents bottom-up --------------------------------
			// Paired XOR (XOR1A/XOR1B): contributions propagate through the tree;
			// when branches merge, only the max reflected current per group counts.
			// Rail-exclusive XOR (XOR3, no letter): resolved locally at the parent;
			// total = max(rail-exclusive load, sum of all other loads on the rail).
			function calculateCurrent(node) {
				if (node.children.length === 0) return;

				var normalSum = 0;
				var xorMerged = {};
				var railXorSum = 0;
				var hasRailXor = false;

				for (var i = 0; i < node.children.length; i++) {
					var child = node.children[i];

					if (isNaN(child.current)) {
						calculateCurrent(child);
					}
					if (isNaN(child.current)) return; // incomplete data

					// Reflection ratio for voltage-converting children
					var ratio = 1;
					if (['sw_reg', 'acdc'].indexOf(child.type) >= 0) {
						if (isNaN(child.vout) || isNaN(node.vout) ||
							node.vout === 0 || child.efficiency === 0) {
							return;
						}
						ratio = child.vout / (child.efficiency * node.vout);
					}

					if (child.railXor) {
						// Rail-exclusive: set aside for comparison against rest of rail
						railXorSum += Math.abs(child.current * ratio);
						hasRailXor = true;
					} else {
						// Normal + paired XOR accumulation
						normalSum += Math.abs(child.normalCurrent * ratio);

						for (var g in child.xorCurrents) {
							if (child.xorCurrents.hasOwnProperty(g)) {
								var reflected = Math.abs(child.xorCurrents[g] * ratio);
								if (!xorMerged.hasOwnProperty(g) || reflected > xorMerged[g]) {
									xorMerged[g] = reflected;
								}
							}
						}
					}
				}

				if (hasRailXor) {
					// Rail-exclusive: max(rail-exclusive total, everything else)
					var otherTotal = normalSum;
					for (var g in xorMerged) {
						if (xorMerged.hasOwnProperty(g)) otherTotal += xorMerged[g];
					}
					var total = Math.max(railXorSum, otherTotal);
					// After resolution, propagate as normal current
					node.normalCurrent = total;
					node.xorCurrents = {};
					node.current = total;
				} else {
					// Standard paired + global XOR logic
					node.normalCurrent = normalSum;
					node.xorCurrents = xorMerged;
					var totalLoad = normalSum;
					var globalXor = 0;
					var hasGlobal = xorMerged.hasOwnProperty('_global');
					for (var g in xorMerged) {
						if (xorMerged.hasOwnProperty(g)) {
							if (g === '_global') {
								globalXor = xorMerged[g];
							} else {
								totalLoad += xorMerged[g];
							}
						}
					}
					if (hasGlobal) {
						// Global-exclusive: max(global load, everything else)
						totalLoad = Math.max(globalXor, totalLoad);
					}
					node.current = totalLoad;
				}
			}

			// Run from root nodes (those with no supplier)
			for (var rid in nodes) {
				if (!nodes.hasOwnProperty(rid)) continue;
				if (!nodes[rid].supplierId) {
					propagateVout(nodes[rid]);
					calculateCurrent(nodes[rid]);
				}
			}

			// --- Write results back ------------------------------------------
			var changed = false;
			model.beginUpdate();
			try {
				for (var wid in nodes) {
					if (!nodes.hasOwnProperty(wid)) continue;
					var n = nodes[wid];
					var c = components[wid];

					// Update load_current for non-load components
					if (n.type !== 'load' && !isNaN(n.current)) {
						if (writeProperty(c, 'load_current', formatCurrent(n.current))) {
							changed = true;
						}
					}

					// Update vout for acdc components
					if (n.type === 'acdc' && !isNaN(n.vout)) {
						if (writeProperty(c, 'vout', formatVoltage(n.vout))) {
							changed = true;
						}
					}
				}
			} finally {
				model.endUpdate();
			}

			if (changed) {
				graph.refresh();
			}
		} finally {
			isRecalculating = false;
		}
	}

	// -------------------------------------------------------------------------
	// Event wiring
	// -------------------------------------------------------------------------

	function debouncedRecalculate() {
		if (isRecalculating) return; // ignore changes from our own writes
		if (recalcTimer) clearTimeout(recalcTimer);
		recalcTimer = setTimeout(recalculate, 500);
	}

	// Recalculate whenever the model changes
	model.addListener(mxEvent.CHANGE, function() {
		debouncedRecalculate();
	});

	// -------------------------------------------------------------------------
	// Menu integration
	// -------------------------------------------------------------------------

	ui.actions.addAction('recalculateCurrents', function() {
		recalculate();
	});

	// Add "Recalculate Currents" to the Extras menu
	var extrasMenu = ui.menus.get('extras');
	if (extrasMenu) {
		var origFunct = extrasMenu.funct;
		extrasMenu.funct = function(menu, parent) {
			origFunct.apply(this, arguments);
			menu.addSeparator(parent);
			menu.addItem('Recalculate Currents', null,
				ui.actions.get('recalculateCurrents').funct, parent);
		};
	}

	// Run once on load
	setTimeout(recalculate, 1000);

	console.log('[PowerBlockCalculator] Plugin loaded.');
});
