import os
import sys
import subprocess
import filecmp
import shutil
import unittest

class TestDataUtilsMethods(unittest.TestCase):
    def is_dir_equal(self, dir1, dir2):
        compared = filecmp.dircmp(dir1, dir2)
        if (compared.left_only or compared.right_only or compared.diff_files 
            or compared.funny_files):
            return False
        for subdir in compared.common_dirs:
            if not self.is_dir_equal(os.path.join(dir1, subdir), os.path.join(dir2, subdir)):
                return False
        return True 

    def test_split_wave_jajp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        print(cur_dir)
        wave_dir = os.path.join(cur_dir, r"test_data\split_wave\data\wave")
        align_dir = os.path.join(cur_dir, r"test_data\split_wave\data\align_lab")
        script_dir = os.path.join(cur_dir, r"test_data\split_wave\data\script")
        out_dir = os.path.join(cur_dir, r"test_data\split_wave\out")
        reg_file = os.path.join(cur_dir, r"data_utils_align_reg.json")

        args =  "-mode split " \
                "-wave " + wave_dir + " " \
                "-lab " + align_dir + " " \
                "-script " + script_dir + " " \
                "-out " + out_dir + " " \
                "-reg_file " + reg_file + " " \
                "-maxtime 3 " \
                "-reg_type eva" 

        if not os.path.isdir(r'.\test_data\split_wave\out'):
            os.makedirs(r'.\test_data\split_wave\out')

        script_path = os.path.join(cur_dir, 'data_utils.py')

        print(r"python.exe {} {}".format(script_path, args))

        p = subprocess.Popen(r"python.exe {} {}".format(script_path, args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)     
        with p.stdout:
            for line in iter(p.stdout.readline, b''): # b'\n'-separated lines
                print(line)

        exitcode = p.wait()

        print("Compare the test info file:")
        expect_file = os.path.join(cur_dir, r"test_data\split_wave\expect")      
        out_file= os.path.join(cur_dir, r"test_data\split_wave\out")
        
        self.assertFalse(self.is_dir_equal(expect_file, out_file))
        shutil.rmtree(r'.\test_data\split_wave\out')

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataUtilsMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)