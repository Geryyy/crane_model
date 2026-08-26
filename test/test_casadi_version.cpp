// CasADi is one pinned version, on both sides of this image.
//
// The OCP port of `docs/features/cbs-ocp-python/` turns on regenerating a
// checked-in C tree byte-identically on a no-op run, and CasADi's code
// generation is version-sensitive; until issue 075 retires the C++ symbolic
// graph, the C++ graph and the Python export also have to agree to 1e-9.
// Neither claim survives a toolchain that is whatever `casadi/casadi` `master`
// happened to be the day a Docker layer was built, so
// `.devcontainer/Dockerfile.vscode` pins a release tag and this file is the
// tripwire on the *running* environment.
//
// It is a tripwire and not a fix.  Rebuilding the image is a human action and
// nothing here can do it.
//
// The two tests separate two failures that have different causes and different
// remedies, and that a single version comparison would report as one thing:
//
//   1.  *Two* CasADis.  A second install ahead of this one on `sys.path` gives
//       the Python export a different library from the one `crane_model` links,
//       and no rebuild helps -- the extra install has to go.  This is asserted
//       against the git revision as well as the version string, because two
//       builds of the same release share a version and cannot share a revision.
//   2.  *One* CasADi, but not the pinned one.  That is a stale image, and the
//       remedy is a rebuild.
//
// Case 2 is skipped rather than failed while the image still carries a `master`
// build, because `CASADI_IS_RELEASE` identifies that state exactly and an agent
// cannot rebuild the container it is running in.  See the notes on issue 067:
// the pin is not in effect until the human rebuilds, and case 1 is the half
// that can be asserted meanwhile.

#include <gtest/gtest.h>

#include <casadi/config.h>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <casadi/core/casadi_meta.hpp>

namespace
{
// The tag `.devcontainer/Dockerfile.vscode` pins, handed down by CMake so that
// the CMakeLists and this file do not each carry a copy of it.
constexpr char kPinnedVersion[] = CRANE_MODEL_CASADI_PINNED_VERSION;

// The four readings the Python half can give about itself, in one interpreter
// start: `__version__` is what a consumer sees, the two `CasadiMeta` calls are
// what the library it loaded says about itself, and `__file__` is which install
// `import casadi` resolved to.
constexpr const char * kPythonProbe =
  "python3 -c 'import casadi; print(casadi.__version__); "
  "print(casadi.CasadiMeta.version()); print(casadi.CasadiMeta.git_revision()); "
  "print(casadi.__file__)'";

// stdout of `command`, one entry per line, trailing newlines stripped.
std::vector<std::string> capture_lines(const std::string & command)
{
  std::vector<std::string> lines;
  FILE * pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return lines;
  }
  std::string output;
  char buffer[512];
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  pclose(pipe);

  std::istringstream stream(output);
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lines.push_back(line);
  }
  return lines;
}

struct PythonCasadi
{
  std::string version;       // casadi.__version__
  std::string meta_version;  // casadi.CasadiMeta.version()
  std::string git_revision;  // casadi.CasadiMeta.git_revision()
  std::string module_file;   // casadi.__file__
};

// Reads the Python half, or fails the calling test with whatever the interpreter
// said instead.  An environment where `import casadi` does not work is not a
// pass: the export script of issue 070 runs on exactly this interpreter.
::testing::AssertionResult read_python_casadi(PythonCasadi * out)
{
  const std::vector<std::string> lines = capture_lines(std::string(kPythonProbe) + " 2>/dev/null");
  if (lines.size() != 4) {
    const std::vector<std::string> diagnostic =
      capture_lines(std::string(kPythonProbe) + " 2>&1");
    std::ostringstream message;
    message << "`import casadi` did not report four values. python3 said:\n";
    for (const std::string & line : diagnostic) {
      message << "  " << line << "\n";
    }
    return ::testing::AssertionFailure() << message.str();
  }
  out->version = lines[0];
  out->meta_version = lines[1];
  out->git_revision = lines[2];
  out->module_file = lines[3];
  return ::testing::AssertionSuccess();
}

// Every reading, gathered into one block, so that whichever comparison fails the
// message carries all of them and not only the two it compared.
std::string context(const PythonCasadi & python)
{
  std::ostringstream out;
  out << "\n"
      << "  pinned tag (Dockerfile.vscode)   : " << kPinnedVersion << "\n"
      << "  C++  CASADI_VERSION_STRING       : " << CASADI_VERSION_STRING << "\n"
      << "  C++  CasadiMeta::version()       : " << casadi::CasadiMeta::version() << "\n"
      << "  C++  CASADI_GIT_REVISION         : " << CASADI_GIT_REVISION << "\n"
      << "  C++  CASADI_GIT_DESCRIBE         : " << CASADI_GIT_DESCRIBE << "\n"
      << "  C++  CASADI_IS_RELEASE           : " << CASADI_IS_RELEASE << "\n"
      << "  py   casadi.__version__          : " << python.version << "\n"
      << "  py   CasadiMeta.version()        : " << python.meta_version << "\n"
      << "  py   CasadiMeta.git_revision()   : " << python.git_revision << "\n"
      << "  py   casadi.__file__             : " << python.module_file << "\n";
  return out.str();
}
}  // namespace

// Case 1.  This holds in every image, pinned or not, and is the half that says
// there is no second CasADi: the version the C++ header declares, the version
// the library `crane_model` links reports at runtime, and the version the Python
// bindings report are one build, identified by its git revision.  A revision is
// unique to a build in a way a release number is not, so two installs of the
// same release still fail here.
TEST(CraneModelCasadiVersion, TheCppAndPythonHalvesAreOneCasadiBuild)
{
  PythonCasadi python;
  ASSERT_TRUE(read_python_casadi(&python));
  const std::string readings = context(python);

  // The header this package compiles against and the library it links.  These
  // differ when `/usr/local/include` and `/usr/local/lib` come from two installs.
  EXPECT_EQ(std::string(CASADI_VERSION_STRING), casadi::CasadiMeta::version())
    << "The CasADi header and the linked CasADi library are different builds." << readings;
  EXPECT_EQ(std::string(CASADI_GIT_REVISION), casadi::CasadiMeta::git_revision())
    << "The CasADi header and the linked CasADi library are different builds." << readings;

  // ... and the library `import casadi` resolved to.
  EXPECT_EQ(casadi::CasadiMeta::version(), python.meta_version)
    << "`import casadi` resolved to a different CasADi than the one C++ links. Remove the "
       "second install; a rebuild will not help." << readings;
  EXPECT_EQ(casadi::CasadiMeta::git_revision(), python.git_revision)
    << "`import casadi` resolved to a different CasADi than the one C++ links. Remove the "
       "second install; a rebuild will not help." << readings;
}

// Case 2.  The version both sides report is the pinned release tag.
//
// `casadi.__version__` is not `CasadiMeta.version()`: off a release tag CasADi
// appends `+` to the version string, and `casadi/__init__.py` then replaces
// `__version__` with `git describe --first-parent`, which on `master` names the
// last tag reachable from it and not the version that was built.  That is why one
// unpinned 3.7.2+ build reported itself to C++ as `3.7.2+` and to Python as
// `3.6.3-1247.f0e2e2b22`, and it is the second reason the pin is a release tag
// and not a commit: at a tag `CASADI_IS_RELEASE` is 1, there is no `+`, the
// substitution does not happen, and the two strings are equal by construction.
TEST(CraneModelCasadiVersion, BothSidesReportThePinnedReleaseTag)
{
  PythonCasadi python;
  ASSERT_TRUE(read_python_casadi(&python));
  const std::string readings = context(python);

  if (CASADI_IS_RELEASE == 0) {
    // The gtest in this image drops the message of a `GTEST_SKIP()` from both
    // the console and the JUnit XML, and a skip nobody can read is not the loud
    // signal this file is for. Print it first.
    std::cout << "\nCasADi is NOT the pinned release in this container.\n"
              << "It was built from an unpinned `master` clone, so the pin in\n"
              << "`.devcontainer/Dockerfile.vscode` is not in effect here and CasADi code\n"
              << "generation in this container is not reproducible. Rebuild the devcontainer\n"
              << "image; it cannot be fixed from inside the container.\n"
              << readings << std::endl;
    GTEST_SKIP() << "CasADi is an unpinned `master` build; rebuild the devcontainer image.";
  }

  EXPECT_EQ(std::string(CASADI_VERSION_STRING), python.version)
    << "The C++ and Python halves report different CasADi versions." << readings;
  EXPECT_EQ(std::string(CASADI_VERSION_STRING), kPinnedVersion)
    << "The C++ CasADi is not the pinned release." << readings;
  EXPECT_EQ(python.version, kPinnedVersion)
    << "The Python CasADi is not the pinned release." << readings;
}
