from xml.etree import ElementTree as ET
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import re

root_types = ['sw_reg','ldo','acdc','source','load','filter','isum']

xml_files = [f for f in os.listdir() if f.endswith('xml')]
if len(xml_files) == 1:
    xml_file = xml_files[0]
else:
    xml_file = filedialog.askopenfilename(title='Select draw.io XML export file.')

# Helper function to extract attributes from an element
def extract_attrs(element):
    obj_id = element.attrib.get('id', None)
    obj_type = element.attrib.get('type', None)
    label = element.attrib.get('label', None)
    parent = element.attrib.get('parent', None)
    source = element.attrib.get('source', None)
    target = element.attrib.get('target', None)
    return obj_id, obj_type, label, parent, source, target

def get_df_from_xml(root):

    # Initialize an empty list to store the data
    data_combined = []

    # Traverse the XML tree and collect data considering both standalone and nested mxCells
    for elem in root.iter():
        if elem.tag == 'object' or elem.tag == 'mxCell':
            obj_id, obj_type, label, parent, source, target = extract_attrs(elem)

            # For nested mxCell inside object, update the parent, source, and target attributes
            if elem.tag == 'object':
                mxCell = elem.find('./mxCell')
                if mxCell is not None:
                    _, _, _, parent, source, target = extract_attrs(mxCell)

            if obj_id or obj_type or label or parent or source or target:  # Skip if all are None
                data_combined.append([elem.tag, obj_id, obj_type, label, parent, source, target])

    # Create a DataFrame considering both standalone and nested mxCells
    return pd.DataFrame(data_combined, columns=['Tag','OID', 'Type', 'Label', 'Parent', 'SrcID', 'DstID'])

# Load the XML file and parse it
tree = ET.parse(xml_file)
root = tree.getroot()

df = get_df_from_xml(root)

# Function to remove HTML tags from a string
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', str(text))

# Apply the function to remove HTML tags from all string columns
for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].apply(remove_html_tags)

# Gets all parents
def get_lineage(df, oid, lin=None):
    if lin is None:
        lin = []
    
    pid_df = df.loc[(df.OID == oid) & (~df.OID.isin([None,'None','1'])), 'Parent'] # Parent ID
    if len(pid_df) == 1:
        pid = pid_df.iloc[0]
        if pid in [None, 'None', '1']:
            return lin
        
        lin += [pid]
        lin = get_lineage(df, pid, lin=lin)
    elif len(pid_df) > 1:
        print(f'Warning: {oid} contains multiple parent objects')
        print(pid_df)
        
    return lin

def flatten_group_attributes(orig_df):
    df = orig_df.copy()

    group_obj_ids = df.loc[df.Type.isin(root_types), 'OID'].tolist()

    for oid in group_obj_ids:

        for i, r in df.loc[(df.Parent == oid) & (~df.OID.isna()) & (df.OID != 'None')].iterrows():
            col_name, col_val = r.Type, r.Label
            
            if col_name in [None, 'None']:

                if col_val not in [None, 'None']:

                    df.loc[df.OID==oid, 'Label'] = col_val

            else:
                df.loc[df.OID==oid, col_name] = col_val
            
    return df.loc[df.Type.isin(root_types)]

obj_df = flatten_group_attributes(df)

def get_root_id_map():
    """
    Many of the drawio objects are created as a group, so I want to create a map from any child object to it's
    group base object.
    """
    id_dict = {}
    for oid in df.OID:
        if oid in [None, 'None', 0, 1, '0', '1']:
            continue
        
        lineage = get_lineage(df, oid)

        if len(lineage) > 0:
            id_dict.update({l: lineage[-1] for l in lineage + [oid] if lineage[-1] != l})
        else:
            id_dict.update({oid: oid})
            
    return id_dict

id_to_root_dict = get_root_id_map()

def get_supply_id_dict():
    """
    Create a dictionary that maps any object to the object ID of the supply rail SW/LDO/etc.
    """
    supply_id_dict = {}

    for i, r in df.loc[(df.SrcID != 'None') & (df.DstID != 'None')].iterrows():
        src, dst = r.SrcID, r.DstID

        supply_id_dict.update({id_to_root_dict.get(dst): id_to_root_dict.get(src)})
        
    return supply_id_dict

supply_id_dict = get_supply_id_dict()

# Update dataframe with supply rail IDs
obj_df.SrcID = [supply_id_dict.get(oid) for oid in obj_df.OID]

def extract_and_convert(s):
    try:
        
        if type(s) == str:
            # Extract only the numbers and the decimal point
            number = re.findall(r"[-+]?\d*\.\d+|\d+", s)

            if not number:
                return s

            value = float(number[0])
            
            if s.startswith('-'):
                value = -value
            

            # Check for units and apply conversion
            if s.endswith('mA'):
                value /= 1e3
            elif s.endswith('uA'):
                value /= 1e6
            elif s.endswith('mV'):
                value /= 1e3
            elif s.endswith('%'):
                value /= 1e2
            elif 'V' in s:
                
                # Handle special voltage formats like '1V2', '0V85'
                if s.endswith('AC'):
                    parts = s[:-len('AC')].split('V')
                else:
                    parts = s.split('V')

                if len(parts) == 2 and parts[1]:
                    value = float(parts[0]) + float(parts[1]) / 10 ** len(parts[1])

            return value
        else:
            return s
        
        return s
    except ValueError:
        return s

def col_to_float(df, col):
    df[col] = df[col].apply(extract_and_convert)
    return df

val_cols = ['vout','current_limit','load_current','efficiency']

# Create an efficiency column if it doesn't exist. It may not exist if all regs are LDOs
if 'efficiency' not in obj_df.columns:
    obj_df['efficiency'] = np.nan

for c in val_cols:
    obj_df = col_to_float(obj_df, c)

# Force non numeric's to NaN
obj_df.load_current = pd.to_numeric(obj_df['load_current'], errors='coerce')
obj_df.vout = pd.to_numeric(obj_df.vout, errors='coerce')

def get_load_multiplier(s):
    match = re.search(r'x(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        return 1
    
def update_loads_with_multiplier(df):

    for i, r in df.loc[(~df.load_name.isna()) & (df.Type == 'load')].iterrows():

        multiplier = get_load_multiplier(r.load_name)
        df.at[i, 'load_current'] = multiplier*r.load_current
        
update_loads_with_multiplier(obj_df)

def get_invalid_rows(df, cols=['Type','Label','load_name','load_current','vout','efficiency','display_name']):
    cond = (df.Type == 'load') & (df.load_current.isna())
    filt = df.loc[cond, cols]
    if len(filt) > 0:
        print('----------------------- MISSING Load Current[s] -----------------------')
        print(filt)
        print()

    cond = (df.Type.isin(['sw_reg','ldo','source'])) & (df.vout.isna())
    filt = df.loc[cond, cols]
    if len(filt) > 0:
        print('----------------------- MISSING Output Voltage[s] -----------------------')
        print(filt)
        print()

    cond = (df.Type != 'source') & ((df.SrcID.isna()) | (df.SrcID.str.lower() == 'none'))
    cols = ['Type','Label','load_name','OID','Parent','SrcID','DstID']
    filt = df.loc[cond, cols]
    if len(filt) > 0:
        print('----------------------- MISSING Source Connection[s] -----------------------')
        print(filt)
        print()

get_invalid_rows(obj_df)

def check_missing_values(df):
    errors = []
    
    for i, r in df.loc[(df.Type == 'load') & df.load_current.isna()].iterrows():
        errors += [f'Add Value to load_current property in Power Block Diagram:\nType={r.Type}\nLabel={r.Label}\nID={r.OID}']
    
    for i, r in df.loc[df.Type.isin(['sw_reg','ldo','source','pwr_sw']) & df.vout.isna()].iterrows():
        errors += [f'Add Value to Vout property in Power Block Diagram: \nType={r.Type}\nLabel={r.Label}\nID={r.OID}']
        
    if len(errors) > 0:
        raise ValueError('\n'.join(errors))
    
check_missing_values(obj_df)

class Node:
    def __init__(self, r):
        self.id = r.OID
        self.type = r.Type
        self.srcid = r.SrcID
        self.current = r.load_current
        self.vout = r.vout
        self.children = []
        
        if type(r.efficiency) == float and not(np.isnan(r.efficiency)):
            self.efficiency = r.efficiency
        else:
            self.efficiency = 1
    
    def add_child(self, node):        
        self.children.append(node)
        
    def set_rectified_voltage(self, vac, forward_voltage_drop=1):
        """
        Calculate the rectified DC voltage from the VAC rms input
        vac: rms AC voltage
        forward_voltage_drop: forward voltage drop across both diodes in full bridge rectifier
        """
        if type(self.vout) != float or np.isnan(self.vout):
            self.vout = vac * 2**0.5 - forward_voltage_drop
            
    def get_vout(self):

        for n in self.children:

            if type(self.vout) == float and not(np.isnan(self.vout)):

                # Pass voltage unchanged
                if n.type in ['pwr_sw','filter','isum']:
                    n.vout = self.vout

                # Calculate DC Rectified output
                elif n.type == 'acdc':
                    n.set_rectified_voltage(self.vout)

            n.get_vout()
                
        
    def get_load_current(self):
        
        total_load = 0
        
        if len(self.children) > 0:
            
            for ch in self.children:
                
                # Complete missing current calcs before continuing
                if np.isnan(ch.current):
                    ch.get_load_current()

                if ch.type in ['sw_reg', 'acdc']:
                    load_current = (ch.current * ch.vout) / (ch.efficiency * self.vout)
                else:
                    load_current = ch.current
                    
                total_load += abs(load_current)
                        
            self.current = total_load   

def check_missing_values(df):
    assert len(df.loc[(df.Type == 'load') & df.load_current.isna()]) == 0, 'Missing load current'
    assert len(df.loc[df.Type.isin(['sw_reg','ldo','source','pwr_sw']) & df.vout.isna()]) == 0, 'Missing output voltage definition'
    
check_missing_values(obj_df)  
            
def calculate_missing_values(df):
    # 1. Create Graph where nodes are objects such as LDO, SW, load, etc. and edges are rails connecting the objects

    graph = {}

    # Create nodes
    for i, r in df.iterrows():
        graph[r.OID] = Node(r)

    # Add nodes
    for oid in graph:
        for dstid in df.loc[df.SrcID == oid, 'OID']:
            graph[oid].add_child(graph[dstid])
            
    # 2. Recursively walk graph and calculate all missing currents first 

    # Calculate load currents starting with highest level node
    for oid in df.loc[df.SrcID.isin([None, 'None']), 'OID']:
        graph[oid].get_vout()
        graph[oid].get_load_current()

    # 3. Move graph current calculations to dataframe
    df['load_current'] = [graph[oid].current for oid in df.OID]
    df['vout'] = [graph[oid].vout for oid in df.OID]
    
calculate_missing_values(obj_df)

# Function to update label by id. Updates xml.
def update_label_by_id(root, update_dict):
    for element in root.iter():
        element_id = element.attrib.get('id')
        
        if element_id in update_dict:
            element.attrib['label'] = update_dict[element_id]

def get_group_obj_id_by_type(parent_id, obj_type='load_current'):
    group_ids = [k for k, v in id_to_root_dict.items() if v == parent_id]
    
    ids = df.loc[df.OID.isin(group_ids) & (df.Type == obj_type), 'OID'].tolist()
    if len(ids) > 1:
        print('Warning! More than one object ID found for specific type.')
        return None
        
    return ids

def get_load_update_dict(df):
    update_dict = {} # Stores the ID, NewLabel pairs
    for i, r in df.loc[df.Type != 'load'].iterrows():
        lbl_ids = get_group_obj_id_by_type(r.OID, 'load_current')
        current = r.load_current

        # Uncomment to help with troubleshooting
        # if pd.isna(current):
        #     cols = ['Type','Label','load_name','load_current','vout','efficiency','display_name']
        #     cols = ['Type','Label','OID','Parent','SrcID','DstID']
        #     print(df.columns)
        #     print(df[[c for c in df.columns if c not in ['Tag','efficiency','current_limit','OID','SrcID']]])

        lbl = f'{round(current, 2)}A' if current >= 0.1 else f'{int(current*1000)}mA'

        for lbl_id in lbl_ids:
            update_dict[lbl_id] = lbl
            
    return update_dict

def get_acdc_update_dict(df):
    update_dict = {} # Stores the ID, NewLabel pairs
    for i, r in df.loc[df.Type == 'acdc'].iterrows():
        lbl_ids = get_group_obj_id_by_type(r.OID, 'vout')
        lbl = f'{round(r.vout, 2)}V' if r.vout >= 0.1 else f'{int(r.vout*1000)}mV'

        for lbl_id in lbl_ids:
            update_dict[lbl_id] = lbl
            
    return update_dict
            
load_update_dict = get_load_update_dict(obj_df)
acdc_update_dict = get_acdc_update_dict(obj_df)

# Parse the XML content from the file
with open(xml_file, 'r') as file:
    xml_content = file.read()
   
# Update Load current in XML
update_label_by_id(root, load_update_dict)
update_label_by_id(root, acdc_update_dict)

# Create new xml file save name
base_name = os.path.splitext(xml_file)[0]  # Removes the extension from the file name
new_file_name = base_name + "_with_currents.xml"  # Appends '_with_currents' and re-adds the '.xml' extension

# Write the modified XML back to a new file if the update was successful
with open(new_file_name, 'wb') as file:
    tree = ET.ElementTree(root)
    tree.write(file)