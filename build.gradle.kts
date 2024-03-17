import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import org.jetbrains.kotlin.js.inline.clean.removeUnusedImports

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

plugins {
    java
    kotlin("jvm") version "1.9.23"
    application
    id("me.champeau.jmh") version "0.7.2"

    id("com.diffplug.spotless") version "6.23.3"
}

group = "org.onheap"

version = "1.0-SNAPSHOT"

repositories { mavenCentral() }

dependencies {
    testImplementation(kotlin("test"))
    implementation("org.apache.logging.log4j:log4j-slf4j-impl:2.22.1")

    implementation("org.jetbrains.kotlinx:multik-core:0.2.2")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.2")
    implementation("org.ejml:ejml-all:0.43.1")

    // DJL dependencies
    implementation(platform("ai.djl:bom:0.25.0"))
    implementation("ai.djl:api")
    implementation("ai.djl.pytorch:pytorch-engine")
    implementation("ai.djl.pytorch:pytorch-native-cpu::osx-aarch64")
    implementation("ai.djl.pytorch:pytorch-jni")
}

tasks.test {
    jvmArgs(listOf("--add-modules", "jdk.incubator.vector"))
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "21"
    kotlinOptions.freeCompilerArgs = listOf("-Xadd-modules=jdk.incubator.vector")
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.withType<JavaExec> { jvmArgs(listOf("--add-modules", "jdk.incubator.vector")) }

// == Benchmark configs start ==
jmh {
    warmupIterations.set(3) // Default is 10
    iterations.set(3) // Default is 10
    //    warmup.set("10s")      // Default is '10 s'
    //    timeOnIteration.set("10s")  // Default is '10 s'

    includes.addAll("benchmarks.NDArrayImplementationBenchmark")
    //    includes.addAll("benchmarks.BenchmarkLibraries")
}

tasks.jmhRunBytecodeGenerator { jvmArgs.addAll("--add-modules=jdk.incubator.vector") }

// == Benchmark configs end ==

spotless {
    java {
        importOrder()
        removeUnusedImports()

        toggleOffOn() // enables the use of // spotless:off and // spotless:on

        // apply a specific flavor of google-java-format
        googleJavaFormat("1.19.1").aosp().reflowLongStrings().skipJavadocFormatting()
        // fix formatting of type annotations
        formatAnnotations()
    }

    kotlin {
        // by default the target is every '.kt' and '.kts` file in the java sourcesets
        ktfmt().dropboxStyle()

        toggleOffOn() // enables the use of // spotless:off and // spotless:on
    }
    kotlinGradle {
        target("*.gradle.kts") // default target for kotlinGradle
        ktfmt().dropboxStyle()
    }
}

application { mainClass.set("MainKt") }
